"""
map_v14.py
----------
Full SLAM pipeline: LK tracking → ORB keyframes → PnP pose → world map

Normal path  (LK has enough tracks):
    LK optical flow   →  updated (u,v) for each tracked point
    solvePnPRansac    →  camera pose
    (uses stored world 3D pts from when each point was first detected —
     no descriptor matching needed on this path)

Fallback path  (LK loses tracks):
    ORB detectAndCompute  →  fresh features + descriptors
    BFMatcher + ratio test →  match vs last keyframe
    solvePnPRansac         →  relocalize

After every pose update:
    lift (u,v,z)   →  X = (u-cx)*z/fx,  Y = (v-cy)*z/fy,  Z = z
    world transform →  P_world = R @ P_cam + t
    voxel dedup     →  accumulate into growing map

Keyframe triggered when camera moves > KF_TRANS or rotates > KF_ROT_DEG.

Saves map_v14.ply on exit.
Controls → Q to quit
"""

import numpy as np
import cv2
import pyrealsense2 as rs
import open3d as o3d
from collections import deque


# ── Settings ──────────────────────────────────────────────────────────────────
PLAYBACK_FILE = r"recordings\2026-04-12_22-52-46.bag"
SAVE_PLY      = "map_v14.ply"

FRAME_W     = 640
FRAME_H     = 480
FRAME_SKIP  = 2

MAX_TRACK   = 300
MIN_TRACK   = 80
EXCL_RADIUS = 15

LK_WIN      = (21, 21)
LK_LEVELS   = 3
LK_MAX_ERR  = 25.0

MIN_DEPTH_M = 0.3
MAX_DEPTH_M = 4.0
VOXEL_SIZE  = 0.05

MIN_MATCHES = 12
PNP_ITERS   = 100

KF_TRANS      = 0.15   # metres  — new keyframe if moved more than this
KF_ROT_DEG    = 15.0   # degrees — new keyframe if rotated more than this
MIN_KF_FRAMES = 8      # minimum frames between keyframes
MAX_KF        = 5      # local map size

# Camera intrinsics (D435 recording)
FX = 384.327880859375
FY = 384.327880859375
CX = 321.8272705078125
CY = 239.01609802246094
K_MAT = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]], dtype=np.float64)
DIST  = np.zeros((4, 1), dtype=np.float64)


# ── RealSense ─────────────────────────────────────────────────────────────────
pipeline = rs.pipeline()
cfg      = rs.config()
cfg.enable_device_from_file(PLAYBACK_FILE, repeat_playback=False)
profile     = pipeline.start(cfg)
profile.get_device().as_playback().set_real_time(False)
align       = rs.align(rs.stream.color)
depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
spatial     = rs.spatial_filter()
temporal    = rs.temporal_filter()
print(f"[PLAYBACK]  {PLAYBACK_FILE}")


# ── Detectors ─────────────────────────────────────────────────────────────────
fast    = cv2.FastFeatureDetector_create(threshold=10, nonmaxSuppression=True)
orb     = cv2.ORB_create(nfeatures=500)
matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

lk_params = dict(winSize=LK_WIN, maxLevel=LK_LEVELS,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))


# ── Helpers ───────────────────────────────────────────────────────────────────
def depth_at(depth_raw, u, v):
    """Return depth in metres at pixel (u,v), or 0 if invalid."""
    if not (0 <= u < FRAME_W and 0 <= v < FRAME_H):
        return 0.0
    z = depth_raw[v, u] * depth_scale
    return z if MIN_DEPTH_M < z < MAX_DEPTH_M else 0.0


def cam_to_world(u, v, z, pose):
    """Lift pixel + depth to world-frame 3D point using current pose."""
    X = (u - CX) * z / FX
    Y = (v - CY) * z / FY
    return pose[:3, :3] @ np.array([X, Y, z]) + pose[:3, 3]


def _pnp(pts3d, pts2d):
    """solvePnPRansac → 4x4 camera-to-world pose, or None on failure."""
    if len(pts3d) < MIN_MATCHES:
        return None
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        pts3d.astype(np.float64), pts2d.astype(np.float64),
        K_MAT, DIST,
        iterationsCount=PNP_ITERS, reprojectionError=4.0, confidence=0.99)
    if not ok or inliers is None or len(inliers) < MIN_MATCHES:
        return None
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R.T
    T[:3, 3]  = -(R.T @ tvec.ravel())
    return T


def _rot_deg(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


# ── Keyframe ──────────────────────────────────────────────────────────────────
class Keyframe:
    __slots__ = ('pose', 'kp', 'des', 'pts3d')
    def __init__(self, pose, kp, des, pts3d):
        self.pose = pose; self.kp = kp; self.des = des; self.pts3d = pts3d


local_map = deque(maxlen=MAX_KF)


def create_keyframe(gray, depth_raw, pose):
    """ORB detect → lift to 3D → store as keyframe."""
    kps, des = orb.detectAndCompute(gray, None)
    if des is None or len(kps) == 0:
        return None
    pts3d = []
    for k in kps:
        u, v = int(k.pt[0]), int(k.pt[1])
        z = depth_at(depth_raw, u, v)
        if z == 0.0:
            pts3d.append([np.nan, np.nan, np.nan])
        else:
            pts3d.append(cam_to_world(u, v, z, pose).tolist())
    kf = Keyframe(pose.copy(), kps, des, np.array(pts3d, dtype=np.float64))
    local_map.append(kf)
    return kf


def orb_fallback(gray):
    """Relocalize: ORB detect → match last KF → PnP."""
    if not local_map:
        return None
    kps, des = orb.detectAndCompute(gray, None)
    if des is None or len(kps) == 0:
        return None
    kf   = local_map[-1]
    raw  = matcher.knnMatch(des, kf.des, k=2)
    good = [m for pair in raw if len(pair) == 2
            for m, n in [(pair[0], pair[1])] if m.distance < 0.75 * n.distance]
    if len(good) < MIN_MATCHES:
        return None
    p3, p2 = [], []
    for m in good:
        pt3 = kf.pts3d[m.trainIdx]
        if np.any(np.isnan(pt3)):
            continue
        p3.append(pt3)
        p2.append(kps[m.queryIdx].pt)
    return _pnp(np.array(p3), np.array(p2))


# ── World map (voxel dedup) ───────────────────────────────────────────────────
_MAP_CAP  = 300_000
_map_buf  = np.zeros((_MAP_CAP, 3), dtype=np.float32)
_map_n    = 0
_occupied = set()


def map_add(pts3d):
    global _map_n
    if len(pts3d) == 0:
        return 0
    idx = np.floor(pts3d / VOXEL_SIZE).astype(np.int32)
    new = [i for i in range(len(idx))
           if (key := (idx[i,0], idx[i,1], idx[i,2])) not in _occupied
           and not _occupied.add(key)]
    if not new:
        return 0
    chunk = pts3d[new].astype(np.float32)
    end   = min(_map_n + len(chunk), _MAP_CAP)
    count = end - _map_n
    _map_buf[_map_n:end] = chunk[:count]
    _map_n = end
    return count


def map_pts():
    return _map_buf[:_map_n]


# ── LK track state ────────────────────────────────────────────────────────────
track_pts2d = np.zeros((0, 1, 2), np.float32)
track_pts3d = np.zeros((0, 3),    np.float64)   # world frame, fixed at detection
prev_gray   = None


def refresh_tracks(gray, depth_raw, pose):
    """Detect new FAST points, lift to world 3D, add to tracker."""
    global track_pts2d, track_pts3d
    mask = np.full((FRAME_H, FRAME_W), 255, dtype=np.uint8)
    for pt in track_pts2d.reshape(-1, 2):
        cv2.circle(mask, (int(pt[0]), int(pt[1])), EXCL_RADIUS, 0, -1)
    kps = fast.detect(gray, mask)
    if not kps:
        return 0
    kps = sorted(kps, key=lambda k: k.response, reverse=True)[:MAX_TRACK]
    new2d, new3d = [], []
    for k in kps:
        u, v = int(k.pt[0]), int(k.pt[1])
        z = depth_at(depth_raw, u, v)
        if z == 0.0:
            continue
        new2d.append([[k.pt[0], k.pt[1]]])
        new3d.append(cam_to_world(u, v, z, pose))
    if not new2d:
        return 0
    n2d = np.array(new2d, np.float32)
    n3d = np.array(new3d, np.float64)
    track_pts2d = np.vstack([track_pts2d, n2d]) if len(track_pts2d) else n2d
    track_pts3d = np.vstack([track_pts3d, n3d]) if len(track_pts3d) else n3d
    if len(track_pts2d) > MAX_TRACK:
        track_pts2d = track_pts2d[:MAX_TRACK]
        track_pts3d = track_pts3d[:MAX_TRACK]
    return len(new2d)


# ── Open3D live window ────────────────────────────────────────────────────────
vis = o3d.visualization.Visualizer()
vis.create_window("map_v14 — world map", width=800, height=600)

pcd = o3d.geometry.PointCloud()
vis.add_geometry(pcd)

trail_line = o3d.geometry.LineSet()   # camera trail
vis.add_geometry(trail_line)

vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3))

vis.get_render_option().background_color = np.array([1.0, 1.0, 1.0])
vis.get_render_option().point_size = 3.0

_o3d_first  = True
_trail_pts  = []   # list of camera positions


def update_o3d(cam_pos):
    global _o3d_first, _trail_pts

    # ── map points ────────────────────────────────────────────────────────────
    pts = map_pts()
    if len(pts) > 0:
        pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
        z = pts[:, 2]
        t = np.clip((z - MIN_DEPTH_M) / (MAX_DEPTH_M - MIN_DEPTH_M), 0, 1)
        # dark blue→dark red on white background
        pcd.colors = o3d.utility.Vector3dVector(
            np.column_stack([t * 0.7, np.zeros(len(t)), (1 - t) * 0.7]))
        vis.update_geometry(pcd)

    # ── camera trail ──────────────────────────────────────────────────────────
    _trail_pts.append(cam_pos.copy())
    if len(_trail_pts) >= 2:
        trail_line.points = o3d.utility.Vector3dVector(np.array(_trail_pts))
        trail_line.lines  = o3d.utility.Vector2iVector(
            [[i, i + 1] for i in range(len(_trail_pts) - 1)])
        trail_line.colors = o3d.utility.Vector3dVector(
            np.tile([1.0, 0.0, 0.0], (len(_trail_pts) - 1, 1)))  # red trail
        vis.update_geometry(trail_line)

    if _o3d_first and len(pts) > 0:
        vis.reset_view_point(True)
        _o3d_first = False


# ── Main loop ─────────────────────────────────────────────────────────────────
global_pose     = np.eye(4)
last_kf_pose    = np.eye(4)
frames_since_kf = 0
frame_idx       = 0
n_inliers       = 0
mode            = "boot"    # boot → tracking → fallback

print("Running — Q to quit\n")

try:
    while True:
        try:
            frames = align.process(pipeline.wait_for_frames(timeout_ms=5000))
        except RuntimeError:
            print("[end of recording]"); break

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            continue

        frame_idx += 1
        if frame_idx % FRAME_SKIP != 0:
            continue

        depth_frame = spatial.process(depth_frame)
        depth_frame = temporal.process(depth_frame)

        color_raw = np.asanyarray(color_frame.get_data())
        depth_raw = np.asanyarray(depth_frame.get_data())
        gray      = cv2.cvtColor(color_raw, cv2.COLOR_RGB2GRAY)
        dbg       = cv2.cvtColor(color_raw, cv2.COLOR_RGB2BGR)

        # ── Seed ──────────────────────────────────────────────────────────────
        if prev_gray is None:
            refresh_tracks(gray, depth_raw, global_pose)
            create_keyframe(gray, depth_raw, global_pose)
            last_kf_pose    = global_pose.copy()
            frames_since_kf = 0
            prev_gray       = gray
            mode            = "tracking"
            print(f"  [seed]  {len(track_pts2d)} tracked pts")
            continue

        # ── LK tracking ───────────────────────────────────────────────────────
        new_pose = None
        if len(track_pts2d) >= MIN_MATCHES:
            next_pts, status, err = cv2.calcOpticalFlowPyrLK(
                prev_gray, gray, track_pts2d, None, **lk_params)
            good = status.ravel() == 1
            if err is not None:
                good &= err.ravel() < LK_MAX_ERR
            track_pts2d = next_pts[good]
            track_pts3d = track_pts3d[good]

            # ── Drop points with no depth at their current position ────────────
            # A tracked point that moved onto a region with z=0 (glass, sky,
            # out-of-range) has no reliable geometry — remove it entirely so it
            # doesn't corrupt PnP or map.
            has_depth = np.array([
                depth_at(depth_raw, int(pt[0]), int(pt[1])) > 0
                for pt in track_pts2d.reshape(-1, 2)
            ])
            track_pts2d = track_pts2d[has_depth]
            track_pts3d = track_pts3d[has_depth]

            # ── PnP on LK correspondences (world 3D ↔ current 2D) ─────────────
            if len(track_pts2d) >= MIN_MATCHES:
                p2 = track_pts2d.reshape(-1, 2).astype(np.float64)
                p3 = track_pts3d.astype(np.float64)
                new_pose = _pnp(p3, p2)
                if new_pose is not None:
                    n_inliers = len(track_pts2d)
                    mode = "tracking"

        # ── ORB fallback: relocalize via descriptor matching ───────────────────
        if new_pose is None:
            new_pose = orb_fallback(gray)
            if new_pose is not None:
                n_inliers = MIN_MATCHES
                mode = "fallback"
                print(f"  [fallback]  frame {frame_idx}")
                track_pts2d = np.zeros((0, 1, 2), np.float32)
                track_pts3d = np.zeros((0, 3),    np.float64)

        # ── Update pose ───────────────────────────────────────────────────────
        if new_pose is not None:
            global_pose = new_pose
        else:
            mode = "lost"

        # ── Re-detect if too few tracks ───────────────────────────────────────
        if len(track_pts2d) < MIN_TRACK:
            refresh_tracks(gray, depth_raw, global_pose)

        # ── Lift current tracked pts → world → add to map ─────────────────────
        # At this point track_pts2d already passed the depth filter above,
        # so every point here is guaranteed to have valid depth.
        if new_pose is not None:
            world_pts = []
            for pt in track_pts2d.reshape(-1, 2):
                u, v = int(pt[0]), int(pt[1])
                z = depth_at(depth_raw, u, v)
                if z > 0:                          # double-check (safety)
                    world_pts.append(cam_to_world(u, v, z, global_pose))
            if world_pts:
                map_add(np.array(world_pts))

        # ── Keyframe check ────────────────────────────────────────────────────
        frames_since_kf += 1
        if frames_since_kf >= MIN_KF_FRAMES:
            dist = np.linalg.norm(global_pose[:3, 3] - last_kf_pose[:3, 3])
            rot  = _rot_deg(global_pose[:3, :3].T @ last_kf_pose[:3, :3])
            if dist > KF_TRANS or rot > KF_ROT_DEG:
                kf = create_keyframe(gray, depth_raw, global_pose)
                if kf:
                    last_kf_pose    = global_pose.copy()
                    frames_since_kf = 0
                    print(f"  [KF]  frame {frame_idx}  map:{_map_n}  "
                          f"dist:{dist:.2f}m  rot:{rot:.1f}°")

        # ── Update Open3D ─────────────────────────────────────────────────────
        update_o3d(global_pose[:3, 3])
        vis.poll_events()
        vis.update_renderer()

        # ── OpenCV overlay ────────────────────────────────────────────────────
        color = (0, 255, 0) if mode == "tracking" else (0, 100, 255)
        for pt in track_pts2d.reshape(-1, 2):
            cv2.circle(dbg, (int(pt[0]), int(pt[1])), 3, color, -1)

        cv2.putText(dbg,
                    f"mode:{mode}  tracked:{len(track_pts2d)}  map:{_map_n}  f:{frame_idx}",
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(dbg,
                    f"pose  x:{global_pose[0,3]:.2f}  y:{global_pose[1,3]:.2f}  z:{global_pose[2,3]:.2f}",
                    (8, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 220, 255), 1)

        cv2.imshow("map_v14 — tracking", dbg)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        prev_gray = gray

finally:
    pipeline.stop()
    vis.destroy_window()
    cv2.destroyAllWindows()

    pts = map_pts()
    if len(pts) > 0:
        out = o3d.geometry.PointCloud()
        out.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
        z = pts[:, 2]
        t = np.clip((z - MIN_DEPTH_M) / (MAX_DEPTH_M - MIN_DEPTH_M), 0, 1)
        out.colors = o3d.utility.Vector3dVector(
            np.column_stack([1 - t, np.zeros(len(t)), t]))
        o3d.io.write_point_cloud(SAVE_PLY, out)
        print(f"\n[saved]  {SAVE_PLY}  —  {len(pts):,} points")
    else:
        print("\n[warn]  no points to save")
    print("[done]")
