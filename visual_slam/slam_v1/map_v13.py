"""
map_v13.py
----------
Feature tracking + depth lift → accumulating world map → saved as .ply

For each tracked (u, v) pixel, reads depth z and computes:
    X = (u - cx) * z / fx
    Y = (v - cy) * z / fy
    Z = z

Points are accumulated into a growing voxel-deduped map each frame.
(No pose yet — points are in camera frame of the first frame.)

On exit saves:  map_v13.ply

Two windows:
  - OpenCV  : RGB feed with tracked feature points
  - Open3D  : live growing 3D map

Controls  →  Q (OpenCV window) to quit
"""

import numpy as np
import cv2
import pyrealsense2 as rs
import open3d as o3d


# ── Settings ──────────────────────────────────────────────────────────────────
PLAYBACK_FILE = r"recordings\2026-04-12_22-52-46.bag"
SAVE_PLY      = "map_v13.ply"

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
VOXEL_SIZE  = 0.05       # dedup grid in metres

# Camera intrinsics (D435 recording)
FX = 384.327880859375
FY = 384.327880859375
CX = 321.8272705078125
CY = 239.01609802246094


# ── RealSense playback ────────────────────────────────────────────────────────
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


# ── Feature tracking ──────────────────────────────────────────────────────────
fast      = cv2.FastFeatureDetector_create(threshold=10, nonmaxSuppression=True)
lk_params = dict(winSize=LK_WIN, maxLevel=LK_LEVELS,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))


def detect_new(gray, existing_pts):
    mask = np.full((FRAME_H, FRAME_W), 255, dtype=np.uint8)
    for pt in existing_pts.reshape(-1, 2):
        cv2.circle(mask, (int(pt[0]), int(pt[1])), EXCL_RADIUS, 0, -1)
    kps = fast.detect(gray, mask)
    if not kps:
        return np.zeros((0, 1, 2), np.float32)
    kps = sorted(kps, key=lambda k: k.response, reverse=True)[:MAX_TRACK]
    return np.array([[k.pt] for k in kps], dtype=np.float32)


# ── Depth lift ────────────────────────────────────────────────────────────────
def lift_3d(pts2d, depth_raw):
    """(u, v) pixel + depth → (X, Y, Z) in metres."""
    pts3d = []
    for pt in pts2d.reshape(-1, 2):
        u, v = int(pt[0]), int(pt[1])
        if not (0 <= u < FRAME_W and 0 <= v < FRAME_H):
            continue
        z = depth_raw[v, u] * depth_scale
        if not (MIN_DEPTH_M < z < MAX_DEPTH_M):
            continue
        X = (u - CX) * z / FX
        Y = (v - CY) * z / FY
        pts3d.append([X, Y, z])
    return np.array(pts3d, dtype=np.float64) if pts3d else np.zeros((0, 3))


# ── Accumulating world map ────────────────────────────────────────────────────
_MAP_CAP  = 200_000
_map_buf  = np.zeros((_MAP_CAP, 3), dtype=np.float32)
_map_n    = 0
_occupied = set()          # voxel keys for dedup


def map_add(pts3d):
    """Add new 3D points, skipping any voxel already occupied."""
    global _map_n
    if len(pts3d) == 0:
        return 0
    idx = np.floor(pts3d / VOXEL_SIZE).astype(np.int32)
    new = []
    for i in range(len(idx)):
        key = (idx[i, 0], idx[i, 1], idx[i, 2])
        if key not in _occupied:
            _occupied.add(key)
            new.append(i)
    if not new:
        return 0
    pts_new = pts3d[new].astype(np.float32)
    end   = min(_map_n + len(pts_new), _MAP_CAP)
    count = end - _map_n
    _map_buf[_map_n:end] = pts_new[:count]
    _map_n = end
    return count


def map_pts():
    return _map_buf[:_map_n]


# ── Open3D live window ────────────────────────────────────────────────────────
vis = o3d.visualization.Visualizer()
vis.create_window("map_v13 — world map", width=800, height=600)

pcd = o3d.geometry.PointCloud()
vis.add_geometry(pcd)
vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3))

opt = vis.get_render_option()
opt.background_color = np.array([0.05, 0.05, 0.05])
opt.point_size = 3.0

_o3d_first = True    # first update needs reset_bounding_box


def update_o3d():
    global _o3d_first
    pts = map_pts()
    if len(pts) == 0:
        return
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    # colour by depth (Z axis): near=red, far=blue
    z = pts[:, 2]
    t = np.clip((z - MIN_DEPTH_M) / (MAX_DEPTH_M - MIN_DEPTH_M), 0, 1)
    pcd.colors = o3d.utility.Vector3dVector(
        np.column_stack([1 - t, np.zeros(len(t)), t]))
    vis.update_geometry(pcd)
    if _o3d_first:
        vis.reset_view_point(True)
        _o3d_first = False


# ── State ─────────────────────────────────────────────────────────────────────
prev_gray = None
track_pts = np.zeros((0, 1, 2), np.float32)
frame_idx = 0

print("Running — Q to quit\n")

try:
    while True:
        try:
            frames = align.process(pipeline.wait_for_frames(timeout_ms=5000))
        except RuntimeError:
            print("[end of recording]")
            break

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

        new_pts = np.zeros((0, 1, 2), np.float32)

        # ── Seed ──────────────────────────────────────────────────────────────
        if prev_gray is None:
            track_pts = detect_new(gray, np.zeros((0, 1, 2), np.float32))
            prev_gray = gray
            print(f"  [seed]  {len(track_pts)} points")
            continue

        # ── LK track ──────────────────────────────────────────────────────────
        if len(track_pts) > 0:
            next_pts, status, err = cv2.calcOpticalFlowPyrLK(
                prev_gray, gray, track_pts, None, **lk_params)
            good = status.ravel() == 1
            if err is not None:
                good &= err.ravel() < LK_MAX_ERR
            track_pts = next_pts[good]

        # ── Re-detect ─────────────────────────────────────────────────────────
        if len(track_pts) < MIN_TRACK:
            new_pts   = detect_new(gray, track_pts)
            track_pts = np.vstack([track_pts, new_pts]) if len(track_pts) else new_pts
            if len(track_pts) > MAX_TRACK:
                track_pts = track_pts[:MAX_TRACK]

        # ── Lift → accumulate → update 3D map ─────────────────────────────────
        pts3d = lift_3d(track_pts, depth_raw)
        added = map_add(pts3d)
        update_o3d()
        vis.poll_events()
        vis.update_renderer()

        # ── OpenCV overlay ────────────────────────────────────────────────────
        for pt in track_pts.reshape(-1, 2):
            cv2.circle(dbg, (int(pt[0]), int(pt[1])), 3, (0, 255, 0), -1)
        for pt in new_pts.reshape(-1, 2):
            cv2.circle(dbg, (int(pt[0]), int(pt[1])), 4, (0, 0, 255), -1)

        cv2.putText(dbg,
                    f"tracked:{len(track_pts)}  map:{_map_n}  added:{added}  f:{frame_idx}",
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        cv2.imshow("map_v13 — tracking", dbg)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        prev_gray = gray

finally:
    pipeline.stop()
    vis.destroy_window()
    cv2.destroyAllWindows()

    # ── Save ──────────────────────────────────────────────────────────────────
    pts = map_pts()
    if len(pts) > 0:
        out = o3d.geometry.PointCloud()
        out.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
        # colour by depth for the saved file too
        z = pts[:, 2]
        t = np.clip((z - MIN_DEPTH_M) / (MAX_DEPTH_M - MIN_DEPTH_M), 0, 1)
        out.colors = o3d.utility.Vector3dVector(
            np.column_stack([1 - t, np.zeros(len(t)), t]))
        o3d.io.write_point_cloud(SAVE_PLY, out)
        print(f"\n[saved]  {SAVE_PLY}  —  {len(pts):,} points")
    else:
        print("\n[warn]  no points to save")

    print("[done]")
