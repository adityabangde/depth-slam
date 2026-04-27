"""
compare_bags.py
---------------
Test if two recordings of the same path A→B produce overlapping trails.

Phase 1  BAG_1 → build ORB map + trail_1  (red)
Phase 2  BAG_2 → localise against that map → trail_2  (blue)
         Map is READ-ONLY in phase 2 — pure localisation test.

Final Open3D window shows both trails overlaid.
If they overlap → method works.  If not → something is wrong.

Press Q in the OpenCV window to skip to the next phase.
"""

import numpy as np
import cv2
import pyrealsense2 as rs
import open3d as o3d


# ── Config ────────────────────────────────────────────────────────────────────
BAG_1 = r"..\recordings\2026-04-12_15-11-52.bag"
BAG_2 = r"..\recordings\2026-04-23_10-39-34.bag"

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


# ── ORB + matcher (shared across both phases) ─────────────────────────────────
orb     = cv2.ORB_create(nfeatures=1500)
matcher = cv2.BFMatcher(cv2.NORM_HAMMING)


# ── Shared map (built in phase 1, read-only in phase 2) ───────────────────────
_map_pts3d  = []
_map_des    = []
_map_voxels = set()


def add_map_point(pos, des):
    key = tuple(np.floor(pos / VOXEL_SIZE).astype(int))
    if key in _map_voxels or len(_map_pts3d) >= MAX_MAP_PTS:
        return False
    _map_voxels.add(key)
    _map_pts3d.append(pos.copy())
    _map_des.append(des.copy())
    return True


# ── Helpers ───────────────────────────────────────────────────────────────────
def depth_at(depth_raw, u, v, win=3):
    u0, v0 = int(u), int(v)
    patch = depth_raw[max(0, v0-win):min(FRAME_H, v0+win+1),
                      max(0, u0-win):min(FRAME_W, u0+win+1)].astype(np.float32) * depth_scale
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


def open_bag(path):
    """Open a bag file, return (pipeline, depth_scale)."""
    global depth_scale
    p   = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device_from_file(path, repeat_playback=False)
    prf = p.start(cfg)
    prf.get_device().as_playback().set_real_time(False)
    ds = prf.get_device().first_depth_sensor().get_depth_scale()
    if not (0.0001 < ds < 0.01):
        ds = 0.001
    depth_scale = ds
    print(f"  depth_scale={ds:.6f}")
    return p


def run_bag(bag_path, phase_label, trail_color_bgr, allow_map_growth):
    """
    Run one bag file.  Returns list of camera positions.
    allow_map_growth=True  → adds new map points (phase 1)
    allow_map_growth=False → read-only map, pure localisation (phase 2)
    """
    print(f"\n{'─'*60}")
    print(f"  {phase_label}  —  {bag_path}")
    print(f"  map_growth={allow_map_growth}   map size at start={len(_map_pts3d)}")
    print(f"  Press Q to skip to next phase")
    print(f"{'─'*60}")

    pipeline  = open_bag(bag_path)
    align     = rs.align(rs.stream.color)
    spatial   = rs.spatial_filter()
    temporal  = rs.temporal_filter()

    global_pose     = np.eye(4)
    last_kf_pose    = np.eye(4)
    frames_since_kf = 0
    frame_idx       = 0
    seeded          = False
    mode            = "boot"
    trail           = []

    try:
        while True:
            try:
                raw = pipeline.wait_for_frames(timeout_ms=5000)
            except RuntimeError:
                print("  [end of file]"); break

            try:
                frames = align.process(raw)
            except RuntimeError:
                continue

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

            # ── Seed first frame ──────────────────────────────────────────────
            if not seeded:
                if allow_map_growth and des is not None:
                    for i, k in enumerate(kps):
                        z = depth_at(depth_raw, k.pt[0], k.pt[1])
                        if z > 0:
                            add_map_point(lift_to_world(k.pt[0], k.pt[1], z, global_pose), des[i])
                last_kf_pose = global_pose.copy()
                seeded = True
                print(f"  [seed]  map:{len(_map_pts3d)}  kps:{len(kps) if kps else 0}")
                continue

            # ── Match against map ─────────────────────────────────────────────
            hamming_thresh = int(MAX_HAMMING * 1.5) if mode == "lost" else MAX_HAMMING
            min_match_req  = max(6, MIN_MATCHES // 2) if mode == "lost" else MIN_MATCHES

            matched_3d, matched_2d, matched_query = [], [], set()

            if des is not None and len(_map_des) >= min_match_req:
                map_des_arr = np.array(_map_des, dtype=np.uint8)
                raw_m = matcher.knnMatch(des, map_des_arr, k=2)
                for pair in raw_m:
                    if len(pair) < 2:
                        continue
                    m, n = pair[0], pair[1]
                    if m.distance < 0.75 * n.distance and m.distance < hamming_thresh:
                        matched_3d.append(_map_pts3d[m.trainIdx])
                        matched_2d.append(kps[m.queryIdx].pt)
                        matched_query.add(m.queryIdx)

            # ── PnP ───────────────────────────────────────────────────────────
            new_pose = pnp(np.array(matched_3d), np.array(matched_2d)) \
                       if len(matched_3d) >= min_match_req else None

            if new_pose is not None:
                global_pose = new_pose
                mode = "map"
            else:
                mode = "lost"

            trail.append(global_pose[:3, 3].copy())

            # ── Keyframe + map extension (phase 1 only) ───────────────────────
            frames_since_kf += 1
            if allow_map_growth and mode == "map" and frames_since_kf >= MIN_KF_FRAMES:
                dist = np.linalg.norm(global_pose[:3, 3] - last_kf_pose[:3, 3])
                rrot = rot_deg(global_pose[:3, :3].T @ last_kf_pose[:3, :3])
                if dist > KF_TRANS or rrot > KF_ROT_DEG:
                    added = 0
                    for i, k in enumerate(kps):
                        if i in matched_query:
                            continue
                        z = depth_at(depth_raw, k.pt[0], k.pt[1])
                        if z > 0:
                            if add_map_point(lift_to_world(k.pt[0], k.pt[1], z, global_pose), des[i]):
                                added += 1
                    last_kf_pose    = global_pose.copy()
                    frames_since_kf = 0
                    print(f"  [KF]  f:{frame_idx}  map:{len(_map_pts3d)}  "
                          f"+{added}  matched:{len(matched_3d)}")

            # ── OpenCV overlay ────────────────────────────────────────────────
            dot_color = (0, 200, 0) if mode == "map" else (0, 0, 220)
            if kps:
                for k in kps:
                    cv2.circle(dbg, (int(k.pt[0]), int(k.pt[1])), 2, dot_color, -1)

            cv2.putText(dbg,
                        f"{phase_label}  mode:{mode}  map:{len(_map_pts3d)}"
                        f"  matched:{len(matched_3d)}  f:{frame_idx}",
                        (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1)
            cv2.putText(dbg,
                        f"x:{global_pose[0,3]:.2f}  y:{global_pose[1,3]:.2f}"
                        f"  z:{global_pose[2,3]:.2f}",
                        (8, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 220, 255), 1)

            cv2.imshow("compare_bags", dbg)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("  [skipped]")
                break

    finally:
        pipeline.stop()

    print(f"  trail length: {len(trail)} poses")
    return trail


# ── Run both phases ───────────────────────────────────────────────────────────
depth_scale = 0.001   # updated inside open_bag()

trail1 = run_bag(BAG_1, "PHASE-1 (build map)", (0, 0, 255), allow_map_growth=True)
trail2 = run_bag(BAG_2, "PHASE-2 (localise)", (255, 0, 0), allow_map_growth=False)

cv2.destroyAllWindows()

np.save("trail_compare_1.npy", np.array(trail1))
np.save("trail_compare_2.npy", np.array(trail2))
print(f"\n[saved]  trail_compare_1.npy ({len(trail1)} pts)  trail_compare_2.npy ({len(trail2)} pts)")

if _map_pts3d:
    pts = np.array(_map_pts3d, dtype=np.float64)
    map_pcd = o3d.geometry.PointCloud()
    map_pcd.points = o3d.utility.Vector3dVector(pts)
    map_pcd.paint_uniform_color([0.6, 0.6, 0.6])
    o3d.io.write_point_cloud("map_compare.ply", map_pcd)
    print(f"[saved]  map_compare.ply  ({len(_map_pts3d):,} points)")


# ── Smoothing ─────────────────────────────────────────────────────────────────
SMOOTH_WIN = 11

def smooth(pts, w):
    pts = np.array(pts, dtype=np.float64)
    out = np.zeros_like(pts)
    for i in range(len(pts)):
        lo = max(0, i - w // 2)
        hi = min(len(pts), i + w // 2 + 1)
        out[i] = pts[lo:hi].mean(axis=0)
    return out


# ── Final Open3D comparison view ──────────────────────────────────────────────
print("\nOpening Open3D comparison window …  (close window to exit)")

vis = o3d.visualization.Visualizer()
vis.create_window("Trail Comparison — RED=bag1  BLUE=bag2", width=1100, height=700)

# map point cloud
if _map_pts3d:
    pts = np.array(_map_pts3d, dtype=np.float64)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.paint_uniform_color([0.7, 0.7, 0.7])
    vis.add_geometry(pcd)

def make_trail_lineset(trail, color):
    if len(trail) < 2:
        return None
    pts = smooth(trail, SMOOTH_WIN)
    ls  = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(pts)
    ls.lines  = o3d.utility.Vector2iVector([[i, i+1] for i in range(len(pts)-1)])
    ls.colors = o3d.utility.Vector3dVector(np.tile(color, (len(pts)-1, 1)))
    return ls

ls1 = make_trail_lineset(trail1, [1.0, 0.0, 0.0])   # red
ls2 = make_trail_lineset(trail2, [0.0, 0.4, 1.0])   # blue

if ls1: vis.add_geometry(ls1)
if ls2: vis.add_geometry(ls2)

# start spheres
for trail, col in [(trail1, [1.0, 0.5, 0.0]), (trail2, [0.0, 0.8, 0.0])]:
    if trail:
        sp = o3d.geometry.TriangleMesh.create_sphere(radius=0.08)
        sp.translate(trail[0])
        sp.paint_uniform_color(col)
        vis.add_geometry(sp)

vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3))

opt = vis.get_render_option()
opt.background_color = np.array([1.0, 1.0, 1.0])
opt.point_size = 2.0
opt.line_width = 3.0

vis.reset_view_point(True)
vis.run()
vis.destroy_window()
print("[done]")
