"""
map_v18.py
----------
Study: ORB feature detection quality + depth → 3-D world map.

Windows:
  [OpenCV]  2-D color frame with keypoints + match lines + HUD charts
  [Open3D]  3-D scatter of current frame's ORB features lifted by depth
              green  = matched to previous frame
              blue→red = unmatched (gradient by response strength)

Per-frame HUD stats: total kp, kp with valid depth, matched, match%, FPS
"""

import time
import numpy as np
import cv2
import pyrealsense2 as rs
import open3d as o3d

# ── Config ────────────────────────────────────────────────────────────────────
PLAYBACK_FILE = r"..\recordings\2026-04-12_22-52-46.bag"

TARGET_FPS  = 20          # display cap
ORB_NFEAT   = 1500        # max features to detect
HISTORY_LEN = 60          # frames to show in the bar chart
MATCH_RATIO = 0.85        # Lowe ratio-test threshold
MAX_HAMMING = 80          # hard Hamming cap after ratio test

MIN_DEPTH_M = 0.3
MAX_DEPTH_M = 12.0

# RealSense D435 intrinsics
FX, FY = 384.327880859375, 384.327880859375
CX, CY = 321.8272705078125, 239.01609802246094
FRAME_W, FRAME_H = 640, 480
K_MAT = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]], dtype=np.float64)
DIST  = np.zeros((4, 1), dtype=np.float64)
MIN_PNP       = 6
KF_TRANS_M    = 0.15    # add keyframe every 15 cm
KF_ROT_DEG    = 10.0    # or 10° rotation
MIN_KF_FRAMES = 5       # minimum frames between keyframes

# ── RealSense ─────────────────────────────────────────────────────────────────
pipeline = rs.pipeline()
cfg      = rs.config()
cfg.enable_device_from_file(PLAYBACK_FILE, repeat_playback=False)
profile  = pipeline.start(cfg)
profile.get_device().as_playback().set_real_time(False)
align    = rs.align(rs.stream.color)

depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
if not (0.0001 < depth_scale < 0.01):
    print(f"[warn] depth_scale={depth_scale} looks wrong → forcing 0.001")
    depth_scale = 0.001
print(f"[depth scale]  {depth_scale:.6f}")

print(f"[PLAYBACK]  {PLAYBACK_FILE}")
print("Press Q in the OpenCV window to quit\n")

# ── ORB + matcher ─────────────────────────────────────────────────────────────
orb     = cv2.ORB_create(nfeatures=ORB_NFEAT)
matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

# ── State ─────────────────────────────────────────────────────────────────────
prev_gray = None
prev_des  = None
prev_kp   = None

kp_history    = []   # (frame_idx, n_keypoints)
match_history = []   # (frame_idx, n_matches)

frame_idx   = 0

# ── Persistent feature map (points never removed once added) ──────────────────
_map_pts    = {}          # pt_id → np.array([x,y,z], float64)  WORLD coords
_kp_to_pid  = {}          # prev-frame kp_idx → pt_id
_next_pid   = 0
camera_pose      = np.eye(4)   # world_from_camera: frame-1 = identity (world origin)

# ── Keyframe database (Step A + B) ───────────────────────────────────────────
_keyframes       = []          # list of dicts per keyframe
_db_des_flat     = None        # (N_total, 32) uint8 — all KF descriptors stacked
_db_pids_flat    = []          # pid per row in _db_des_flat  (-1 = no map point)
_flann           = None        # FlannBasedMatcher (LSH for binary descriptors)
_last_kf_pose    = np.eye(4)
_frames_since_kf = 0
_kf_positions    = []          # world xyz per keyframe (for visualisation)


# ── Open3D world-map window ───────────────────────────────────────────────────
vis3d = o3d.visualization.Visualizer()
vis3d.create_window("map_v18 — 3-D feature world map", width=800, height=600)

feat_pcd   = o3d.geometry.PointCloud()
trail_line = o3d.geometry.LineSet()
kf_pcd     = o3d.geometry.PointCloud()   # keyframe position markers
vis3d.add_geometry(feat_pcd)
vis3d.add_geometry(trail_line)
vis3d.add_geometry(kf_pcd)
vis3d.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3))

opt3d = vis3d.get_render_option()
opt3d.background_color = np.array([1.0, 1.0, 1.0])
opt3d.point_size = 5.0

_o3d_inited  = False
_cam_trail   = []   # list of camera positions in display coords (y-flipped)


# ── Helpers ───────────────────────────────────────────────────────────────────
def depth_at(depth_raw, u, v, win=3):
    """Median depth (metres) in a win×win patch; returns 0 if invalid."""
    u0, v0 = int(u), int(v)
    patch = depth_raw[max(0, v0 - win):min(FRAME_H, v0 + win + 1),
                      max(0, u0 - win):min(FRAME_W, u0 + win + 1)].astype(np.float32) * depth_scale
    valid = patch[(patch > MIN_DEPTH_M) & (patch < MAX_DEPTH_M)]
    return float(np.median(valid)) if len(valid) >= 2 else 0.0


def unproject(u, v, z):
    """Pixel + depth → 3-D point in camera space (x right, y down, z forward)."""
    return np.array([(u - CX) * z / FX,
                     (v - CY) * z / FY,
                      z], dtype=np.float64)


def solve_pose(pts3d_world, pts2d_curr):
    """
    PnP: world-coord map points + their current pixels → world_from_camera 4×4.
    pts3d_world : (N,3) float64   pts2d_curr : (N,2) float64
    Returns 4×4 SE3 or None if PnP fails.
    """
    if len(pts3d_world) < MIN_PNP:
        return None
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        pts3d_world, pts2d_curr, K_MAT, DIST,
        iterationsCount=100, reprojectionError=4.0, confidence=0.99)
    if not ok or inliers is None or len(inliers) < MIN_PNP:
        return None
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R.T
    T[:3, 3]  = -(R.T @ tvec.ravel())   # camera centre in world
    return T


def pose_delta(p1, p2):
    """Translation (m) and rotation (deg) between two SE3 poses."""
    t = float(np.linalg.norm(p1[:3, 3] - p2[:3, 3]))
    R = p1[:3, :3].T @ p2[:3, :3]
    a = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0))))
    return t, a


def db_add_keyframe(fid, pose, des_arr, kp_list, kp_to_pid_map):
    """Snapshot current frame as a keyframe and rebuild the FLANN index."""
    global _db_des_flat, _db_pids_flat, _flann
    if des_arr is None or len(des_arr) == 0:
        return
    pids = [kp_to_pid_map.get(i, -1) for i in range(len(kp_list))]
    _keyframes.append({'id': len(_keyframes), 'frame': fid,
                       'pose': pose.copy(), 'des': des_arr.copy(), 'pids': pids})
    _kf_positions.append(pose[:3, 3].copy())

    all_des  = np.vstack([kf['des']  for kf in _keyframes]).astype(np.uint8)
    all_pids = []
    for kf in _keyframes:
        all_pids.extend(kf['pids'])
    _db_des_flat  = all_des
    _db_pids_flat = all_pids

    _flann = cv2.FlannBasedMatcher(
        dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1),
        dict(checks=50))
    _flann.add([all_des])
    _flann.train()
    print(f"  [KF {len(_keyframes):03d}]  frame={fid:04d}"
          f"  db_rows={len(all_pids)}  map_pts={len(_map_pts)}")


def kp_color(response, low, high):
    """Blue (weak) → green → red (strong) by response magnitude."""
    t = np.clip((response - low) / max(high - low, 1e-9), 0.0, 1.0)
    r = int(t * 255)
    b = int((1 - t) * 255)
    return (b, 100, r)   # BGR


def draw_bar_chart(canvas, history, x, y, w, h, color, label):
    """Draw a mini bar chart for the last HISTORY_LEN values in history."""
    if not history:
        return
    vals = [v for _, v in history[-HISTORY_LEN:]]
    mx   = max(vals) if max(vals) > 0 else 1
    bw   = max(1, w // len(vals))
    # background
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (30, 30, 30), -1)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (80, 80, 80),  1)
    for i, v in enumerate(vals):
        bh = int(v / mx * (h - 4))
        bx = x + i * bw
        cv2.rectangle(canvas, (bx, y + h - bh - 2), (bx + bw - 1, y + h - 2), color, -1)
    cv2.putText(canvas, label, (x + 4, y + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
    cv2.putText(canvas, f"max {mx}", (x + w - 55, y + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)


# ── Main loop ─────────────────────────────────────────────────────────────────
while True:
    t_frame_start = time.perf_counter()

    # ── grab frame ────────────────────────────────────────────────────────────
    try:
        frames = pipeline.wait_for_frames(timeout_ms=5000)
    except RuntimeError:
        print("[INFO]  End of bag.")
        break

    aligned = align.process(frames)
    color_f = aligned.get_color_frame()
    depth_f = aligned.get_depth_frame()
    if not color_f or not depth_f:
        continue

    color_img = np.asanyarray(color_f.get_data())   # H×W×3 RGB → convert below
    color_img = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)
    depth_raw = np.asanyarray(depth_f.get_data())   # H×W uint16, raw units
    gray      = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)

    frame_idx += 1

    # ── ORB detection ─────────────────────────────────────────────────────────
    kp, des = orb.detectAndCompute(gray, None)
    n_kp    = len(kp) if kp else 0

    kp_history.append((frame_idx, n_kp))

    # ── Match against previous frame ──────────────────────────────────────────
    n_matches    = 0
    good_matches = []

    if prev_des is not None and des is not None and len(des) >= 2:
        raw = matcher.knnMatch(des, prev_des, k=2)
        for pair in raw:
            if len(pair) < 2:
                continue
            a, b = pair
            if a.distance < MATCH_RATIO * b.distance and a.distance < MAX_HAMMING:
                good_matches.append(a)
        n_matches = len(good_matches)

    match_history.append((frame_idx, n_matches))

    # ── Update world map ─────────────────────────────────────────────────────
    new_kp_to_pid = {}
    matched_pids  = set()
    new_pids      = set()
    pnp_pts3d     = []
    pnp_pts2d     = []

    # ── Step A: FLANN global match against keyframe database ─────────────────
    flann_kp_to_pid = {}
    n_flann = 0
    if _flann is not None and des is not None and len(des) >= 2:
        try:
            raw_f = _flann.knnMatch(des.astype(np.uint8), k=2)
            for pair in raw_f:
                if len(pair) < 2:
                    continue
                m, n_ = pair
                if (m.distance < MATCH_RATIO * n_.distance
                        and m.distance < MAX_HAMMING
                        and m.queryIdx not in flann_kp_to_pid):
                    pid = _db_pids_flat[m.trainIdx]
                    if pid != -1 and pid in _map_pts:
                        flann_kp_to_pid[m.queryIdx] = pid
            n_flann = len(flann_kp_to_pid)
        except Exception:
            pass

    # ── Frame-to-frame carry-forward (fills gaps between keyframes) ───────────
    curr_to_prev = {m.queryIdx: m.trainIdx for m in good_matches}
    ft_kp_to_pid = {}
    if kp:
        for i in range(len(kp)):
            if i in curr_to_prev:
                pid = _kp_to_pid.get(curr_to_prev[i])
                if pid is not None and pid in _map_pts:
                    ft_kp_to_pid[i] = pid

    # Merge: FLANN (global) overrides frame-to-frame (local)
    merged = {**ft_kp_to_pid, **flann_kp_to_pid}

    # ── Collect PnP inputs from merged matches ────────────────────────────────
    if kp:
        for i, k in enumerate(kp):
            if i not in merged:
                continue
            pid = merged[i]
            new_kp_to_pid[i] = pid
            matched_pids.add(pid)
            pnp_pts3d.append(_map_pts[pid])
            pnp_pts2d.append([k.pt[0], k.pt[1]])

    # ── Step B: PnP pose from merged anchors ─────────────────────────────────
    pose_src = "init"
    if len(pnp_pts3d) >= MIN_PNP:
        new_pose = solve_pose(
            np.array(pnp_pts3d, dtype=np.float64),
            np.array(pnp_pts2d, dtype=np.float64))
        if new_pose is not None:
            camera_pose = new_pose
            pose_src = f"flann={n_flann} ft={len(ft_kp_to_pid)}"

    # ── Add new (unmatched) features in world coordinates ─────────────────────
    if kp:
        for i, k in enumerate(kp):
            if i in new_kp_to_pid:
                continue
            z = depth_at(depth_raw, k.pt[0], k.pt[1])
            if z == 0.0:
                continue
            p_cam   = unproject(k.pt[0], k.pt[1], z)
            p_world = camera_pose[:3, :3] @ p_cam + camera_pose[:3, 3]
            pid              = _next_pid
            _map_pts[pid]    = p_world
            new_kp_to_pid[i] = pid
            new_pids.add(pid)
            _next_pid       += 1

    _kp_to_pid = new_kp_to_pid

    # ── Keyframe check (Step A) ───────────────────────────────────────────────
    _frames_since_kf += 1
    t_dist, r_ang = pose_delta(camera_pose, _last_kf_pose)
    if _frames_since_kf >= MIN_KF_FRAMES and (t_dist > KF_TRANS_M or r_ang > KF_ROT_DEG):
        db_add_keyframe(frame_idx, camera_pose, des, kp or [], new_kp_to_pid)
        _last_kf_pose    = camera_pose.copy()
        _frames_since_kf = 0

    # ── Build Open3D cloud (all map points — none ever removed) ───────────────
    pts3d_list  = []
    cols3d_list = []
    for pid, pos in _map_pts.items():
        pts3d_list.append([pos[0], -pos[1], pos[2]])   # y-flip for display
        if pid in matched_pids:
            cols3d_list.append([0.0,  0.70, 0.0 ])     # green  – matched this frame
        elif pid in new_pids:
            cols3d_list.append([0.95, 0.45, 0.0 ])     # orange – new this frame
        else:
            cols3d_list.append([0.60, 0.60, 0.85])     # blue-grey – established

    n_map_pts = len(pts3d_list)

    # ── Camera trail ─────────────────────────────────────────────────────────
    cam_pos = camera_pose[:3, 3].copy()
    cam_pos[1] = -cam_pos[1]           # y-flip to match display coords
    _cam_trail.append(cam_pos)

    if len(_cam_trail) >= 2:
        trail_pts = np.array(_cam_trail, dtype=np.float64)
        trail_line.points = o3d.utility.Vector3dVector(trail_pts)
        trail_line.lines  = o3d.utility.Vector2iVector(
            [[i, i + 1] for i in range(len(_cam_trail) - 1)])
        trail_line.colors = o3d.utility.Vector3dVector(
            np.tile([1.0, 0.0, 0.0], (len(_cam_trail) - 1, 1)))   # red
        vis3d.update_geometry(trail_line)

    # ── Push feature cloud to Open3D ──────────────────────────────────────────
    if n_map_pts > 0:
        feat_pcd.points = o3d.utility.Vector3dVector(np.array(pts3d_list,  dtype=np.float64))
        feat_pcd.colors = o3d.utility.Vector3dVector(np.array(cols3d_list, dtype=np.float64))
        vis3d.update_geometry(feat_pcd)
        if not _o3d_inited:
            vis3d.reset_view_point(True)
            _o3d_inited = True

    # ── Keyframe position markers (blue diamonds) ─────────────────────────────
    if _kf_positions:
        kf_disp = np.array([[p[0], -p[1], p[2]] for p in _kf_positions], dtype=np.float64)
        kf_pcd.points = o3d.utility.Vector3dVector(kf_disp)
        kf_pcd.colors = o3d.utility.Vector3dVector(
            np.tile([0.0, 0.3, 1.0], (len(_kf_positions), 1)))
        vis3d.update_geometry(kf_pcd)

    vis3d.poll_events()
    vis3d.update_renderer()

    # ── Draw keypoints colored by response strength ────────────────────────────
    vis = color_img.copy()

    if kp:
        responses = np.array([k.response for k in kp])
        r_lo2, r_hi2 = float(responses.min()), float(responses.max())
        for k in kp:
            c = kp_color(k.response, r_lo2, r_hi2)
            cv2.circle(vis, (int(k.pt[0]), int(k.pt[1])), 3, c, -1)
            angle_rad = np.deg2rad(k.angle)
            ex = int(k.pt[0] + 6 * np.cos(angle_rad))
            ey = int(k.pt[1] + 6 * np.sin(angle_rad))
            cv2.line(vis, (int(k.pt[0]), int(k.pt[1])), (ex, ey), c, 1)

    # ── Draw matches to previous frame (green lines) ──────────────────────────
    if good_matches and prev_kp is not None:
        for m in good_matches:
            pt_curr = tuple(map(int, kp[m.queryIdx].pt))
            pt_prev = tuple(map(int, prev_kp[m.trainIdx].pt))
            cv2.line(vis, pt_curr, pt_prev, (0, 220, 60), 1)

    # ── HUD overlay ───────────────────────────────────────────────────────────
    H, W = vis.shape[:2]

    # semi-transparent dark bar at top
    overlay = vis.copy()
    cv2.rectangle(overlay, (0, 0), (W, 28), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, vis, 0.45, 0, vis)

    match_pct = (n_matches / n_kp * 100) if n_kp > 0 else 0.0
    hud = (f"Frame {frame_idx:04d}  |  "
           f"KP: {n_kp:4d}  |  "
           f"Map: {n_map_pts:4d}  |  "
           f"KF: {len(_keyframes):3d}  |  "
           f"FLANN: {n_flann:4d}  FT: {len(ft_kp_to_pid):4d}  |  "
           f"New: {len(new_pids):3d}  |  Q=quit")
    cv2.putText(vis, hud, (8, 19),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1, cv2.LINE_AA)

    # ── Mini bar charts (bottom strip) ────────────────────────────────────────
    chart_h = 60
    chart_y = H - chart_h - 4
    draw_bar_chart(vis, kp_history,    4,       chart_y, W // 2 - 8, chart_h,
                   (60, 180, 255), "Keypoints / frame")
    draw_bar_chart(vis, match_history, W // 2 + 4, chart_y, W // 2 - 8, chart_h,
                   (60, 220, 60),  "Matches to prev frame")

    # ── Show ───────────────────────────────────────────────────���────────────────
    cv2.imshow("map_v18 — ORB feature study", vis)

    print(f"  frame {frame_idx:04d}  kp={n_kp:4d}  map={n_map_pts:4d}"
          f"  kf={len(_keyframes):3d}  flann={n_flann:4d}  ft={len(ft_kp_to_pid):4d}"
          f"  new={len(new_pids):3d}  [{pose_src}]")

    key = cv2.waitKey(1) & 0xFF
    if key in (ord('q'), 27):
        break

    # ── Advance state ─────────────────────────────────────────────────────────
    prev_gray = gray
    prev_des  = des
    prev_kp   = kp

# ── Cleanup ───────────────────────────────────────────────────────────────────
pipeline.stop()
cv2.destroyAllWindows()
vis3d.destroy_window()
print(f"\nDone.  Processed {frame_idx} frames.")
if kp_history:
    counts = [v for _, v in kp_history]
    print(f"  Keypoints — mean: {np.mean(counts):.0f}  "
          f"min: {np.min(counts)}  max: {np.max(counts)}")
if match_history:
    mc = [v for _, v in match_history if v > 0]
    if mc:
        print(f"  Matches   — mean: {np.mean(mc):.0f}  "
              f"min: {np.min(mc)}  max: {np.max(mc)}")
