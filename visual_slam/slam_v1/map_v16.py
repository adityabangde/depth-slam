"""
map_v16.py
----------
ORB descriptor matching against global map → PnP pose.

Changes from v15
────────────────
1. ORB nfeatures 500 → 1500  (more matches, more stable PnP)
2. Map points only added when tracking (mode=map), never when lost
   (bad poses from lost mode were polluting the map with wrong 3D positions)

Saves map_v16.ply and trail_v16.npy on exit.   Q to quit.
"""

import numpy as np
import cv2
import pyrealsense2 as rs
import open3d as o3d


# ── Settings ──────────────────────────────────────────────────────────────────
PLAYBACK_FILE = r"recordings\2026-04-23_15-58-17.bag"
SAVE_PLY      = "map_v16.ply"

FRAME_W     = 640
FRAME_H     = 480
FRAME_SKIP  = 2

MIN_DEPTH_M = 0.3
MAX_DEPTH_M = 12.0
VOXEL_SIZE  = 0.05

MIN_MATCHES = 12
MAX_HAMMING = 60

KF_TRANS      = 0.15
KF_ROT_DEG    = 15.0
MIN_KF_FRAMES = 8
MAX_MAP_PTS   = 50_000

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
profile  = pipeline.start(cfg)
profile.get_device().as_playback().set_real_time(False)
align    = rs.align(rs.stream.color)
spatial  = rs.spatial_filter()
temporal = rs.temporal_filter()

depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
if not (0.0001 < depth_scale < 0.01):
    print(f"[warn]  depth_scale={depth_scale} looks wrong → forcing 0.001")
    depth_scale = 0.001
print(f"[PLAYBACK]  {PLAYBACK_FILE}")
print(f"[depth scale]  {depth_scale:.6f}")


# ── ORB + matcher ─────────────────────────────────────────────────────────────
orb     = cv2.ORB_create(nfeatures=1500)       # v16: was 500
matcher = cv2.BFMatcher(cv2.NORM_HAMMING)


# ── Global map ────────────────────────────────────────────────────────────────
_map_pts3d  = []
_map_des    = []
_map_voxels = set()


def add_map_point(pos, des):
    if len(_map_pts3d) >= MAX_MAP_PTS:
        return False
    key = tuple(np.floor(pos / VOXEL_SIZE).astype(int))
    if key in _map_voxels:
        return False
    _map_voxels.add(key)
    _map_pts3d.append(pos.copy())
    _map_des.append(des.copy())
    return True


# ── Helpers ───────────────────────────────────────────────────────────────────
def depth_at(depth_raw, u, v, win=3):
    u0, v0 = int(u), int(v)
    patch = depth_raw[max(0, v0 - win):min(FRAME_H, v0 + win + 1),
                      max(0, u0 - win):min(FRAME_W, u0 + win + 1)].astype(np.float32) * depth_scale
    valid = patch[(patch > MIN_DEPTH_M) & (patch < MAX_DEPTH_M)]
    return float(np.median(valid)) if len(valid) > 0 else 0.0


def lift_to_world(u, v, z, pose):
    p_cam = np.array([(u - CX) * z / FX, (v - CY) * z / FY, z])
    return pose[:3, :3] @ p_cam + pose[:3, 3]


def pnp(pts3d, pts2d):
    if len(pts3d) < MIN_MATCHES:
        return None
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        pts3d.astype(np.float64), pts2d.astype(np.float64),
        K_MAT, DIST, iterationsCount=100, reprojectionError=4.0, confidence=0.99)
    if not ok or inliers is None or len(inliers) < MIN_MATCHES:
        return None
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R.T
    T[:3, 3]  = -(R.T @ tvec.ravel())
    return T


def rot_deg(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


# ── Open3D window ─────────────────────────────────────────────────────────────
vis = o3d.visualization.Visualizer()
vis.create_window("map_v16 — ORB map tracking", width=900, height=650)

pcd        = o3d.geometry.PointCloud()
trail_line = o3d.geometry.LineSet()
vis.add_geometry(pcd)
vis.add_geometry(trail_line)
vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3))

opt = vis.get_render_option()
opt.background_color = np.array([1.0, 1.0, 1.0])
opt.point_size = 3.0

_o3d_first = True
_trail_pts  = []


def update_o3d(cam_pos):
    global _o3d_first, _trail_pts
    _trail_pts.append(cam_pos.copy())

    if _map_pts3d:
        pts = np.array(_map_pts3d, dtype=np.float64)
        pcd.points = o3d.utility.Vector3dVector(pts)
        z = pts[:, 2]
        t = np.clip((z - MIN_DEPTH_M) / (MAX_DEPTH_M - MIN_DEPTH_M), 0, 1)
        pcd.colors = o3d.utility.Vector3dVector(
            np.column_stack([t * 0.7, np.zeros(len(t)), (1 - t) * 0.7]))
        vis.update_geometry(pcd)

    if len(_trail_pts) >= 2:
        trail_line.points = o3d.utility.Vector3dVector(np.array(_trail_pts))
        trail_line.lines  = o3d.utility.Vector2iVector(
            [[i, i + 1] for i in range(len(_trail_pts) - 1)])
        trail_line.colors = o3d.utility.Vector3dVector(
            np.tile([1.0, 0.0, 0.0], (len(_trail_pts) - 1, 1)))
        vis.update_geometry(trail_line)

    if _o3d_first and _map_pts3d:
        vis.reset_view_point(True)
        _o3d_first = False


# ── Main loop ─────────────────────────────────────────────────────────────────
global_pose     = np.eye(4)
last_kf_pose    = np.eye(4)
frames_since_kf = 0
frame_idx       = 0
seeded          = False
mode            = "boot"

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

        kps, des = orb.detectAndCompute(gray, None)

        # ── Seed first frame ──────────────────────────────────────────────────
        if not seeded:
            added = 0
            if des is not None:
                for i, k in enumerate(kps):
                    z = depth_at(depth_raw, k.pt[0], k.pt[1])
                    if z > 0:
                        if add_map_point(lift_to_world(k.pt[0], k.pt[1], z, global_pose),
                                         des[i]):
                            added += 1
            last_kf_pose = global_pose.copy()
            seeded = True
            print(f"  [seed]  {added} map points  ({len(kps)} ORB kps detected)")
            continue

        # ── Match ORB descriptors against global map ──────────────────────────
        hamming_thresh = int(MAX_HAMMING * 1.5) if mode == "lost" else MAX_HAMMING
        min_match_req  = max(6, MIN_MATCHES // 2) if mode == "lost" else MIN_MATCHES

        matched_3d, matched_2d, matched_query = [], [], set()

        if des is not None and len(_map_des) >= min_match_req:
            map_des_arr = np.array(_map_des, dtype=np.uint8)
            raw = matcher.knnMatch(des, map_des_arr, k=2)
            for pair in raw:
                if len(pair) < 2:
                    continue
                m, n = pair[0], pair[1]
                if m.distance < 0.75 * n.distance and m.distance < hamming_thresh:
                    matched_3d.append(_map_pts3d[m.trainIdx])
                    matched_2d.append(kps[m.queryIdx].pt)
                    matched_query.add(m.queryIdx)

        # ── PnP → camera pose ─────────────────────────────────────────────────
        new_pose = pnp(np.array(matched_3d), np.array(matched_2d)) \
                   if len(matched_3d) >= min_match_req else None

        if new_pose is not None:
            global_pose = new_pose
            mode = "map"
        else:
            mode = "lost"

        # ── Keyframe + map extension (only when tracking) ─────────────────────
        frames_since_kf += 1
        if mode == "map" and frames_since_kf >= MIN_KF_FRAMES:
            dist = np.linalg.norm(global_pose[:3, 3] - last_kf_pose[:3, 3])
            rrot = rot_deg(global_pose[:3, :3].T @ last_kf_pose[:3, :3])
            if dist > KF_TRANS or rrot > KF_ROT_DEG:
                added = 0
                for i, k in enumerate(kps):
                    if i in matched_query:
                        continue
                    z = depth_at(depth_raw, k.pt[0], k.pt[1])
                    if z > 0:
                        if add_map_point(lift_to_world(k.pt[0], k.pt[1], z, global_pose),
                                         des[i]):
                            added += 1
                last_kf_pose    = global_pose.copy()
                frames_since_kf = 0
                print(f"  [KF]  f:{frame_idx}  map:{len(_map_pts3d)}  "
                      f"+{added}  matched:{len(matched_3d)}  "
                      f"dist:{dist:.2f}m  rot:{rrot:.1f}°")

        # ── Open3D ────────────────────────────────────────────────────────────
        update_o3d(global_pose[:3, 3])
        vis.poll_events()
        vis.update_renderer()

        # ── OpenCV overlay ────────────────────────────────────────────────────
        dot_color = (0, 200, 0) if mode == "map" else (0, 0, 220)
        if kps:
            for k in kps:
                cv2.circle(dbg, (int(k.pt[0]), int(k.pt[1])), 2, dot_color, -1)

        cv2.putText(dbg,
                    f"mode:{mode}  map:{len(_map_pts3d)}  matched:{len(matched_3d)}  f:{frame_idx}",
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)
        cv2.putText(dbg,
                    f"x:{global_pose[0,3]:.2f}  y:{global_pose[1,3]:.2f}  z:{global_pose[2,3]:.2f}",
                    (8, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 220, 255), 1)

        cv2.imshow("map_v16 — tracking", dbg)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    vis.destroy_window()
    cv2.destroyAllWindows()

    if _trail_pts:
        np.save("trail_v16.npy", np.array(_trail_pts))
        print(f"[saved]  trail_v16.npy  —  {len(_trail_pts)} poses")

    if _map_pts3d:
        pts = np.array(_map_pts3d, dtype=np.float64)
        out = o3d.geometry.PointCloud()
        out.points = o3d.utility.Vector3dVector(pts)
        z = pts[:, 2]
        t = np.clip((z - MIN_DEPTH_M) / (MAX_DEPTH_M - MIN_DEPTH_M), 0, 1)
        out.colors = o3d.utility.Vector3dVector(
            np.column_stack([t * 0.7, np.zeros(len(t)), (1 - t) * 0.7]))
        o3d.io.write_point_cloud(SAVE_PLY, out)
        print(f"\n[saved]  {SAVE_PLY}  —  {len(_map_pts3d):,} points")
    else:
        print("\n[warn]  no points to save")
    print("[done]")
