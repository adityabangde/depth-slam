"""
map_v5.py
---------
map_v4  +  3 ORB-SLAM3-style tracking upgrades.

NEW IN V5
─────────
1. KEYFRAME SELECTION
   Not every frame becomes a keyframe.  A new keyframe is created only when
   the camera has moved / rotated enough, or when too few local map points are
   visible.  This keeps the local map sparse and stable.

2. LOCAL MAP  (last MAX_KF keyframes)
   Each keyframe stores the 3-D world positions of its ORB features.
   Every new frame matches against ALL local map points, not just the previous
   frame.  Shake the camera for 10 frames → local map still has stable
   keyframes from before the shake.

3. GUIDED MATCHING  +  MOTION MODEL
   (a) Constant-velocity motion model predicts the next pose.
   (b) Local map 3-D points are projected into the predicted frame.
   (c) ORB matches are searched only inside a small pixel window around
       each projection — not globally.
   → Far fewer false matches, robust to fast motion.

KEPT FROM V4
────────────
   • Voxel occupancy set  — no double walls
   • Free-space carving   — removes shadow artifacts
   • Depth-only point cloud (colour used only for pose)

Controls  →  Q to quit, saves map_v5.ply
"""

import numpy as np
import cv2
import pyrealsense2 as rs
import open3d as o3d
from collections import deque


# ── Settings ──────────────────────────────────────────────────────────────────
PLAYBACK_FILE = r"recordings\2026-04-12_15-11-52.bag"
                       # set to None to use live camera

FRAME_W       = 640
FRAME_H       = 480
FPS           = 30
MIN_DEPTH_M   = 0.3
MAX_DEPTH_M   = 4.0
VOXEL_SIZE    = 0.05
ORB_FEATURES  = 1000
RENDER_EVERY  = 3
CARVE_EVERY   = 5
FREE_MARGIN   = 0.10

# Keyframe thresholds
MAX_KF        = 8      # local map size (number of keyframes kept)
KF_TRANS      = 0.12   # metres  — add keyframe if moved more than this
KF_ROT_DEG    = 12.0   # degrees — add keyframe if rotated more than this
MIN_KF_FRAMES = 5      # minimum frames between consecutive keyframes

# Guided matching
SEARCH_RADIUS = 30     # pixel search window radius around predicted position
MAX_HAMMING   = 60     # reject match if Hamming distance exceeds this
MIN_MATCHES   = 12     # minimum PnP inliers to accept a pose


# ── Hamming distance LUT (faster than cv2.norm in a tight loop) ───────────────
_POPCOUNT = np.array([bin(i).count('1') for i in range(256)], dtype=np.int32)

def hamming(a: np.ndarray, b: np.ndarray) -> int:
    return int(_POPCOUNT[np.bitwise_xor(a, b)].sum())


# ── RealSense — live camera or recorded .bag file ────────────────────────────
pipeline = rs.pipeline()
cfg      = rs.config()

if PLAYBACK_FILE:
    # Playback: .bag already contains stream configs + intrinsics
    cfg.enable_device_from_file(PLAYBACK_FILE, repeat_playback=False)
    profile = pipeline.start(cfg)
    # Process as fast as possible (don't drop frames to match real-time)
    profile.get_device().as_playback().set_real_time(False)
    print(f"[PLAYBACK]  {PLAYBACK_FILE}")
else:
    cfg.enable_stream(rs.stream.depth, FRAME_W, FRAME_H, rs.format.z16,  FPS)
    cfg.enable_stream(rs.stream.color, FRAME_W, FRAME_H, rs.format.rgb8, FPS)
    profile = pipeline.start(cfg)
    print("[LIVE]  D435 camera")

align = rs.align(rs.stream.color)

depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
intr = (profile.get_stream(rs.stream.color)
               .as_video_stream_profile()
               .get_intrinsics())
fx, fy = intr.fx, intr.fy
cx, cy = intr.ppx, intr.ppy
K    = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], dtype=np.float64)
dist = np.zeros((4,1), dtype=np.float64)

print(f"[D435] fx={fx:.1f}  fy={fy:.1f}  cx={cx:.1f}  cy={cy:.1f}")

spatial  = rs.spatial_filter()
temporal = rs.temporal_filter()


# ── FAST detector + ORB descriptor + BFMatcher ───────────────────────────────
# FAST (Rosten & Drummond 2006) is the original "fast corner" algorithm.
# It just checks a circle of 16 pixels — no pyramid, no orientation scoring.
# We use it for detection only; ORB is called only to compute descriptors
# at those locations, skipping its own (slower) detector step.
fast    = cv2.FastFeatureDetector_create(threshold=10, nonmaxSuppression=True)
orb     = cv2.ORB_create(nfeatures=ORB_FEATURES)
matcher = cv2.BFMatcher(cv2.NORM_HAMMING)


def detect_features(gray: np.ndarray):
    """FAST corners → ORB descriptors at those locations."""
    kp = fast.detect(gray, None)
    if not kp:
        return [], None
    # Sort by corner response, keep the strongest ORB_FEATURES
    kp = sorted(kp, key=lambda k: k.response, reverse=True)[:ORB_FEATURES]
    kp, des = orb.compute(gray, kp)
    return kp, des


# ── Open3D viewer ─────────────────────────────────────────────────────────────
vis = o3d.visualization.Visualizer()
vis.create_window("3D Map v5 — Local Map + Guided", width=1280, height=720)

global_map = o3d.geometry.PointCloud()
vis.add_geometry(global_map)
vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3))

ropt = vis.get_render_option()
ropt.point_size       = 2.0
ropt.background_color = np.array([0.05, 0.05, 0.05])

occupied: set = set()


# ══════════════════════════════════════════════════════════════════════════════
# Keyframe
# ══════════════════════════════════════════════════════════════════════════════
class Keyframe:
    """
    Stores everything needed for local-map matching:
      pose   — cam-to-world 4×4
      kp     — ORB keypoints
      des    — ORB descriptors  (N, 32) uint8
      pts3d  — world-frame 3-D position of each keypoint  (N, 3), NaN if no depth
    """
    __slots__ = ('id', 'pose', 'kp', 'des', 'pts3d')

    def __init__(self, id, pose, kp, des, pts3d):
        self.id    = id
        self.pose  = pose
        self.kp    = kp
        self.des   = des
        self.pts3d = pts3d


local_map: deque = deque(maxlen=MAX_KF)
_kf_id            = 0


def create_keyframe(pose: np.ndarray,
                    gray: np.ndarray,
                    depth_raw: np.ndarray,
                    kp=None, des=None) -> 'Keyframe | None':
    """Pass kp/des if already detected this frame to avoid re-running FAST."""
    global _kf_id
    if kp is None or des is None:
        kp, des = detect_features(gray)
    if des is None or len(kp) == 0:
        return None

    pts3d = np.full((len(kp), 3), np.nan, dtype=np.float64)
    for i, k in enumerate(kp):
        u, v = k.pt
        d    = depth_raw[int(v), int(u)] * depth_scale
        if MIN_DEPTH_M < d < MAX_DEPTH_M:
            pts3d[i] = (pose[:3, :3] @
                        np.array([(u-cx)/fx*d, (v-cy)/fy*d, d])
                        + pose[:3, 3])

    kf      = Keyframe(_kf_id, pose, kp, des, pts3d)
    _kf_id += 1
    return kf


# ══════════════════════════════════════════════════════════════════════════════
# Spatial grid  (fast NN lookup among current-frame keypoints)
# ══════════════════════════════════════════════════════════════════════════════
class KpGrid:
    def __init__(self, kp_list, des_array, cell: int = 30):
        self.cell = cell
        self.grid: dict = {}
        for i, k in enumerate(kp_list):
            gx = int(k.pt[0] / cell)
            gy = int(k.pt[1] / cell)
            self.grid.setdefault((gx, gy), []).append((i, k.pt, des_array[i]))

    def query(self, u: float, v: float, radius: float) -> list:
        gx  = int(u / self.cell)
        gy  = int(v / self.cell)
        nc  = int(radius / self.cell) + 1
        r2  = radius * radius
        out = []
        for dx in range(-nc, nc + 1):
            for dy in range(-nc, nc + 1):
                for item in self.grid.get((gx + dx, gy + dy), []):
                    du = item[1][0] - u
                    dv = item[1][1] - v
                    if du * du + dv * dv <= r2:
                        out.append(item)
        return out


# ══════════════════════════════════════════════════════════════════════════════
# Geometry helpers  (same as v4)
# ══════════════════════════════════════════════════════════════════════════════
def apply_T(pts: np.ndarray, T: np.ndarray) -> np.ndarray:
    return (T[:3, :3] @ pts.T).T + T[:3, 3]


def kp_to_world(kp_list, depth_raw: np.ndarray,
                cam_pose: np.ndarray) -> np.ndarray:
    """
    Project depth at each keypoint location into world frame.

    FAST corners land on edges where depth is often 0 (invalid mixed pixels).
    We search a small window around each keypoint and take the median of valid
    depth values — this recovers depth for the vast majority of features.
    """
    pts  = []
    R, t = cam_pose[:3, :3], cam_pose[:3, 3]
    H, W = depth_raw.shape
    WIN  = 3   # search ±3 pixels around keypoint

    for k in kp_list:
        u0, v0 = int(k.pt[0]), int(k.pt[1])
        # collect valid depths in the window
        u1, u2 = max(0, u0-WIN), min(W, u0+WIN+1)
        v1, v2 = max(0, v0-WIN), min(H, v0+WIN+1)
        patch  = depth_raw[v1:v2, u1:u2].astype(np.float32) * depth_scale
        valid  = patch[(patch > MIN_DEPTH_M) & (patch < MAX_DEPTH_M)]
        if len(valid) == 0:
            continue
        d = float(np.median(valid))
        pts.append(R @ np.array([(k.pt[0]-cx)/fx*d,
                                  (k.pt[1]-cy)/fy*d, d]) + t)
    return np.array(pts, dtype=np.float32) if pts else np.zeros((0, 3), np.float32)


def depth_color(pts: np.ndarray) -> np.ndarray:
    t = np.clip(pts[:, 2] / MAX_DEPTH_M, 0, 1).astype(np.float32)
    return np.stack([1 - t, np.zeros_like(t), t], axis=1)


# ══════════════════════════════════════════════════════════════════════════════
# Map write / carve  (same as v4)
# ══════════════════════════════════════════════════════════════════════════════
def map_add(pts_world: np.ndarray) -> int:
    if len(pts_world) == 0:
        return 0
    idx   = np.floor(pts_world / VOXEL_SIZE).astype(np.int32)
    new_i = []
    for i in range(len(idx)):
        key = (idx[i, 0], idx[i, 1], idx[i, 2])
        if key not in occupied:
            occupied.add(key)
            new_i.append(i)
    if not new_i:
        return 0
    new_pts = pts_world[new_i]
    cur_pts = np.asarray(global_map.points)
    cur_col = np.asarray(global_map.colors)
    if len(cur_pts):
        global_map.points = o3d.utility.Vector3dVector(np.vstack([cur_pts, new_pts]))
        global_map.colors = o3d.utility.Vector3dVector(np.vstack([cur_col, depth_color(new_pts)]))
    else:
        global_map.points = o3d.utility.Vector3dVector(new_pts)
        global_map.colors = o3d.utility.Vector3dVector(depth_color(new_pts))
    return len(new_i)


def carve_free_space(depth_raw: np.ndarray, cam_pose: np.ndarray) -> int:
    pts = np.asarray(global_map.points)
    if len(pts) == 0:
        return 0
    w2c     = np.linalg.inv(cam_pose)
    pts_cam = (w2c[:3, :3] @ pts.T).T + w2c[:3, 3]
    z       = pts_cam[:, 2]
    u       = pts_cam[:, 0] / z * fx + cx
    v       = pts_cam[:, 1] / z * fy + cy
    visible = ((z > MIN_DEPTH_M) &
               (u >= 0) & (u < FRAME_W - 1) &
               (v >= 0) & (v < FRAME_H - 1))
    vis_idx = np.where(visible)[0]
    if len(vis_idx) == 0:
        return 0
    d_meas  = depth_raw[v[vis_idx].astype(int), u[vis_idx].astype(int)].astype(np.float32) * depth_scale
    in_free = (d_meas > MIN_DEPTH_M) & (z[vis_idx] < d_meas - FREE_MARGIN)
    rm_idx  = vis_idx[in_free]
    if len(rm_idx) == 0:
        return 0
    rm_keys = np.floor(pts[rm_idx] / VOXEL_SIZE).astype(int)
    for k in rm_keys:
        occupied.discard((k[0], k[1], k[2]))
    keep = np.ones(len(pts), dtype=bool)
    keep[rm_idx] = False
    global_map.points = o3d.utility.Vector3dVector(pts[keep])
    global_map.colors = o3d.utility.Vector3dVector(np.asarray(global_map.colors)[keep])
    return len(rm_idx)


# ══════════════════════════════════════════════════════════════════════════════
# Pose estimation
# ══════════════════════════════════════════════════════════════════════════════
def _pnp(pts3d: np.ndarray, pts2d: np.ndarray):
    """Run solvePnPRansac. Returns (4×4 cam-to-world, n_inliers) or (None, 0)."""
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        pts3d, pts2d, K, dist,
        iterationsCount=200, reprojectionError=4.0, confidence=0.99,
    )
    if not ok or inliers is None or len(inliers) < MIN_MATCHES:
        return None, 0
    R, _  = cv2.Rodrigues(rvec)
    T     = np.eye(4)
    T[:3, :3] = R.T
    T[:3,  3] = -(R.T @ tvec.ravel())
    return T, len(inliers)


def project_local_map(predicted_pose: np.ndarray):
    """
    Project all local-map 3-D points into the predicted frame.
    Returns (pts3d, des, uv_pred) arrays for points inside the image,
    or three empty arrays if the local map is empty.
    """
    if not local_map:
        return (np.zeros((0, 3)),
                np.zeros((0, 32), dtype=np.uint8),
                np.zeros((0, 2)))

    w2c = np.linalg.inv(predicted_pose)
    R, t = w2c[:3, :3], w2c[:3, 3]

    acc_pts, acc_des, acc_uv = [], [], []

    for kf in local_map:
        valid = ~np.any(np.isnan(kf.pts3d), axis=1)
        if not valid.any():
            continue
        pts = kf.pts3d[valid]           # (M, 3)
        des = kf.des[valid]             # (M, 32)

        pts_cam = (R @ pts.T).T + t
        z       = pts_cam[:, 2]
        front   = z > MIN_DEPTH_M
        if not front.any():
            continue

        u = pts_cam[front, 0] / z[front] * fx + cx
        v = pts_cam[front, 1] / z[front] * fy + cy
        ib = (u >= 0) & (u < FRAME_W) & (v >= 0) & (v < FRAME_H)

        acc_pts.append(pts[front][ib])
        acc_des.append(des[front][ib])
        acc_uv.append(np.stack([u[ib], v[ib]], axis=1))

    if not acc_pts:
        return (np.zeros((0, 3)),
                np.zeros((0, 32), dtype=np.uint8),
                np.zeros((0, 2)))

    return np.vstack(acc_pts), np.vstack(acc_des), np.vstack(acc_uv)


def guided_match(curr_kp, curr_des,
                 map_pts3d, map_des, map_uv):
    """
    For each local-map point projected to (up, vp):
      1. Find current-frame keypoints within SEARCH_RADIUS of (up, vp)
      2. Accept the one with the lowest Hamming distance (if < MAX_HAMMING)
    Returns (pts3d, pts2d) ready for PnP, both float64.
    """
    if curr_des is None or len(map_pts3d) == 0:
        return np.zeros((0, 3), np.float64), np.zeros((0, 2), np.float64)

    grid       = KpGrid(curr_kp, curr_des)
    pts3d_out  = []
    pts2d_out  = []
    used       = set()          # each current keypoint used at most once

    for pt3d, des_map, (up, vp) in zip(map_pts3d, map_des, map_uv):
        candidates = grid.query(up, vp, SEARCH_RADIUS)
        if not candidates:
            continue

        best_d   = MAX_HAMMING
        best_idx = -1
        best_uv  = None

        for idx, pt2d, des_curr in candidates:
            if idx in used:
                continue
            d = hamming(des_map, des_curr)
            if d < best_d:
                best_d   = d
                best_idx = idx
                best_uv  = pt2d

        if best_idx >= 0:
            used.add(best_idx)
            pts3d_out.append(pt3d)
            pts2d_out.append(best_uv)

    if not pts3d_out:
        return np.zeros((0, 3), np.float64), np.zeros((0, 2), np.float64)

    return (np.array(pts3d_out, dtype=np.float64),
            np.array(pts2d_out, dtype=np.float64))


def fallback_pose(curr_kp, curr_des):
    """
    Raw descriptor matching against the most recent keyframe.
    Used when the local map has too few visible points (scene just changed).
    """
    if not local_map or curr_des is None:
        return None, 0
    last_kf = local_map[-1]
    if last_kf.des is None:
        return None, 0

    raw  = matcher.knnMatch(curr_des, last_kf.des, k=2)
    good = [m for pair in raw if len(pair) == 2
            for m, n in [(pair[0], pair[1])]
            if m.distance < 0.75 * n.distance]
    if len(good) < MIN_MATCHES:
        return None, 0

    pts3d, pts2d = [], []
    for m in good:
        p3 = last_kf.pts3d[m.trainIdx]
        if np.any(np.isnan(p3)):
            continue
        pts3d.append(p3)
        pts2d.append(curr_kp[m.queryIdx].pt)
    if len(pts3d) < MIN_MATCHES:
        return None, 0

    return _pnp(np.array(pts3d, np.float64), np.array(pts2d, np.float64))


# ── Keyframe decision ─────────────────────────────────────────────────────────
def _rotation_angle(R_diff: np.ndarray) -> float:
    return float(np.degrees(
        np.arccos(np.clip((np.trace(R_diff) - 1.0) / 2.0, -1.0, 1.0))
    ))


def needs_keyframe(curr_pose: np.ndarray,
                   last_kf_pose: np.ndarray,
                   n_matches: int,
                   frames_since_kf: int) -> bool:
    if frames_since_kf < MIN_KF_FRAMES:
        return False
    t_diff = np.linalg.norm(curr_pose[:3, 3] - last_kf_pose[:3, 3])
    if t_diff > KF_TRANS:
        return True
    R_diff = curr_pose[:3, :3].T @ last_kf_pose[:3, :3]
    if _rotation_angle(R_diff) > KF_ROT_DEG:
        return True
    if n_matches < MIN_MATCHES * 2:   # too few anchors → refresh local map
        return True
    return False


# ══════════════════════════════════════════════════════════════════════════════
# Main loop
# ══════════════════════════════════════════════════════════════════════════════
print("\nMapping — move camera slowly.  Q to quit.\n")

global_pose    = np.eye(4)
velocity       = np.eye(4)     # constant-velocity motion model
last_kf_pose   = np.eye(4)
frames_since_kf = 0
frame_idx      = 0
view_reset     = False
trajectory     = []            # list of (3,) camera positions, one per frame

try:
    while True:

        # ── 1. Capture ────────────────────────────────────────────────────────
        try:
            frames = align.process(pipeline.wait_for_frames(timeout_ms=5000))
        except RuntimeError:
            print("  [end of recording]")
            break
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            continue

        depth_frame = spatial.process(depth_frame)
        depth_frame = temporal.process(depth_frame)

        color_raw = np.asanyarray(color_frame.get_data())
        depth_raw = np.asanyarray(depth_frame.get_data())
        gray      = cv2.cvtColor(color_raw, cv2.COLOR_RGB2GRAY)

        # ── 2. FAST corners + ORB descriptors ────────────────────────────────
        kp, des = detect_features(gray)

        # ── 3. First frame — seed local map ───────────────────────────────────
        if not local_map:
            kf = create_keyframe(global_pose, gray, depth_raw, kp, des)
            if kf:
                local_map.append(kf)
                last_kf_pose    = global_pose.copy()
                frames_since_kf = 0
                pts_world = kp_to_world(kp, depth_raw, global_pose)
                map_add(pts_world)
                vis.update_geometry(global_map)
                print(f"  [KF 0] seeded — {len(global_map.points):,} pts")
            frame_idx += 1
            continue

        # ── 4. Predict pose with motion model ─────────────────────────────────
        predicted_pose = global_pose @ velocity

        # ── 5. Project local map into predicted frame ─────────────────────────
        map_pts3d, map_des, map_uv = project_local_map(predicted_pose)

        # ── 6. Guided matching ────────────────────────────────────────────────
        new_pose   = None
        n_inliers  = 0

        if len(map_pts3d) >= MIN_MATCHES:
            pts3d, pts2d = guided_match(kp, des, map_pts3d, map_des, map_uv)
            if len(pts3d) >= MIN_MATCHES:
                new_pose, n_inliers = _pnp(pts3d, pts2d)

        # ── 7. Fallback: raw match against last keyframe ───────────────────────
        if new_pose is None:
            new_pose, n_inliers = fallback_pose(kp, des)
            if new_pose is not None:
                print(f"  [fallback]  frame {frame_idx}  inliers={n_inliers}")

        # ── 8. Update pose + motion model ─────────────────────────────────────
        if new_pose is not None:
            velocity    = np.linalg.inv(global_pose) @ new_pose
            global_pose = new_pose
        else:
            # Apply predicted pose; decay velocity toward identity
            global_pose = predicted_pose
            velocity    = velocity * 0.5 + np.eye(4) * 0.5
            print(f"  [skip]  frame {frame_idx} — tracking lost, using prediction")

        trajectory.append(global_pose[:3, 3].copy())

        frames_since_kf += 1

        # ── 9. Keyframe decision ───────────────────────────────────────────────
        if needs_keyframe(global_pose, last_kf_pose, n_inliers, frames_since_kf):
            kf = create_keyframe(global_pose, gray, depth_raw, kp, des)
            if kf:
                local_map.append(kf)
                last_kf_pose    = global_pose.copy()
                frames_since_kf = 0
                print(f"  [KF {kf.id}]  frame {frame_idx}  "
                      f"local_map={len(local_map)}  inliers={n_inliers}")

        # ── 10. Free-space carving ────────────────────────────────────────────
        if frame_idx % CARVE_EVERY == 0:
            removed = carve_free_space(depth_raw, global_pose)
            if removed:
                print(f"  [carve]  -{removed} pts")

        # ── 11. Add feature-only map points ───────────────────────────────────
        pts_world = kp_to_world(kp, depth_raw, global_pose)
        added     = map_add(pts_world)

        if frame_idx % RENDER_EVERY == 0:
            vis.update_geometry(global_map)

        if frame_idx % 60 == 0:
            print(f"  [map]  frame {frame_idx:4d} — {len(global_map.points):,} pts")

        # ── 12. Debug preview (CV2 window) ───────────────────────────────────
        dbg = cv2.cvtColor(color_raw, cv2.COLOR_RGB2BGR)
        for k in kp:
            cv2.circle(dbg, (int(k.pt[0]), int(k.pt[1])), 3, (0, 255, 0), -1)
        kf_count = len(local_map)
        map_pts  = len(global_map.points)
        cv2.putText(dbg, f"feats:{len(kp)}  KFs:{kf_count}  map:{map_pts}  "
                         f"f:{frame_idx}",
                    (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 1)
        cv2.putText(dbg, f"inliers:{n_inliers}  added:{added}",
                    (8, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,220,255), 1)
        cv2.imshow("map_v5 — features", dbg)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # ── 13. Render Open3D ─────────────────────────────────────────────────
        if not view_reset:
            vis.reset_view_point(True)
            view_reset = True

        if not vis.poll_events():
            break
        vis.update_renderer()

        frame_idx += 1

finally:
    o3d.io.write_point_cloud("map_v5.ply", global_map)
    if trajectory:
        np.save("map_v5_trajectory.npy", np.array(trajectory))
    print(f"\n[saved] map_v5.ply          — {len(global_map.points):,} points")
    print(f"[saved] map_v5_trajectory.npy — {len(trajectory)} poses")
    pipeline.stop()
    vis.destroy_window()
    cv2.destroyAllWindows()
    print("[done]")
