"""
localizer.py
------------
Phase 2: Load an optimised map (map.npz), then process the same bag file
         frame-by-frame using pure PnP localisation against the loaded map.
         No map growth, no bundle adjustment — read-only map.

Output:  trail_localizer.npy  — (N, 3) camera positions in world space

Run:
    python localizer.py          (after map_builder.py has finished)
"""

import numpy as np
import cv2
import pyrealsense2 as rs
import open3d as o3d


# ── Config ────────────────────────────────────────────────────────────────────
BAG_FILE    = r"..\recordings\2026-04-12_22-52-46.bag"
MAP_FILE    = "map.npz"
TRAIL_OUT   = "trail_localizer.npy"

FRAME_W     = 640
FRAME_H     = 480
FRAME_SKIP  = 2

MIN_DEPTH_M = 0.3
MAX_DEPTH_M = 12.0

MIN_MATCHES = 8
MAX_HAMMING = 80
RATIO_TEST  = 0.85

SMOOTH_WIN  = 11   # centred moving average for display trail

PIXEL_STEP   = 4
VOXEL_DOWN   = 0.05
MAX_ICP_DIST = 0.3
ICP_FITNESS  = 0.25

FX = 384.327880859375
FY = 384.327880859375
CX = 321.8272705078125
CY = 239.01609802246094
K_MAT = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]], dtype=np.float64)
DIST  = np.zeros((4, 1), dtype=np.float64)


# ── Load map ──────────────────────────────────────────────────────────────────
print(f"[map]  Loading {MAP_FILE} …")
data    = np.load(MAP_FILE)
map_pts = data["pts3d"].astype(np.float64)    # (N, 3)
map_des = data["des"].astype(np.uint8)        # (N, 32)
print(f"[map]  {len(map_pts)} points  {len(map_des)} descriptors")


# ── RealSense ─────────────────────────────────────────────────────────────────
pipeline = rs.pipeline()
cfg      = rs.config()
cfg.enable_device_from_file(BAG_FILE, repeat_playback=False)
profile  = pipeline.start(cfg)
profile.get_device().as_playback().set_real_time(False)
align    = rs.align(rs.stream.color)
spatial  = rs.spatial_filter()
temporal = rs.temporal_filter()

depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
if not (0.0001 < depth_scale < 0.01):
    depth_scale = 0.001
print(f"[bag]  {BAG_FILE}")
print(f"[depth scale]  {depth_scale:.6f}\n")


# ── ORB + matcher ─────────────────────────────────────────────────────────────
orb     = cv2.ORB_create(nfeatures=1500)
matcher = cv2.BFMatcher(cv2.NORM_HAMMING)


# ── Helpers ───────────────────────────────────────────────────────────────────
def pnp(pts3d, pts2d, init_pose=None):
    if len(pts3d) < MIN_MATCHES:
        return None
    use_guess = False
    rvec0 = tvec0 = None
    if init_pose is not None:
        R_cw = init_pose[:3, :3].T
        rvec0, _ = cv2.Rodrigues(R_cw)
        tvec0 = (-R_cw @ init_pose[:3, 3]).reshape(3, 1)
        use_guess = True
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        pts3d.astype(np.float64), pts2d.astype(np.float64),
        K_MAT, DIST, rvec0, tvec0, use_guess,
        iterationsCount=200, reprojectionError=4.0, confidence=0.99)
    if not ok or inliers is None or len(inliers) < MIN_MATCHES:
        return None
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R.T
    T[:3, 3]  = -(R.T @ tvec.ravel())
    return T


def depth_to_cloud(depth_raw):
    pts = []
    for v in range(0, FRAME_H, PIXEL_STEP):
        for u in range(0, FRAME_W, PIXEL_STEP):
            z = depth_raw[v, u] * depth_scale
            if MIN_DEPTH_M < z < MAX_DEPTH_M:
                pts.append([(u - CX) * z / FX, (v - CY) * z / FY, z])
    if len(pts) < 50:
        return None
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(pts, dtype=np.float64))
    pcd = pcd.voxel_down_sample(VOXEL_DOWN)
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    return pcd


def smooth_trail(pts, win):
    if len(pts) < win:
        return np.array(pts)
    half = win // 2
    out  = []
    for i in range(len(pts)):
        lo = max(0, i - half)
        hi = min(len(pts), i + half + 1)
        out.append(np.mean(pts[lo:hi], axis=0))
    return np.array(out)


# ── Open3D visualiser ─────────────────────────────────────────────────────────
vis   = o3d.visualization.Visualizer()
vis.create_window("localizer — trail", width=800, height=600)

trail_pcd = o3d.geometry.PointCloud()
map_pcd   = o3d.geometry.PointCloud()

map_pcd.points = o3d.utility.Vector3dVector(map_pts)
map_pcd.paint_uniform_color([0.6, 0.6, 0.6])
vis.add_geometry(map_pcd)
vis.add_geometry(trail_pcd)

o3d_init = False


# ── Main loop ─────────────────────────────────────────────────────────────────
trail_raw  = []
global_pose = None
mode        = "lost"
frame_idx   = 0

prev_cloud = None
prev_T_rel = np.eye(4)
icp_pose   = None

print("Localising … (Q to stop)\n")

try:
    while True:
        try:
            frames = align.process(pipeline.wait_for_frames(timeout_ms=5000))
        except RuntimeError:
            print("[end of bag]"); break

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            continue

        frame_idx += 1
        if frame_idx % FRAME_SKIP != 0:
            continue

        depth_frame = spatial.process(depth_frame)
        depth_frame = temporal.process(depth_frame)

        color_raw  = np.asanyarray(color_frame.get_data())
        depth_raw  = np.asanyarray(depth_frame.get_data())
        gray       = cv2.cvtColor(color_raw, cv2.COLOR_RGB2GRAY)
        dbg        = cv2.cvtColor(color_raw, cv2.COLOR_RGB2BGR)
        curr_cloud = depth_to_cloud(depth_raw)

        kps, des = orb.detectAndCompute(gray, None)

        # ── ICP odometry (frame-to-frame VO) ──────────────────────────────────
        if prev_cloud is not None and curr_cloud is not None:
            icp_result = o3d.pipelines.registration.registration_icp(
                prev_cloud, curr_cloud, MAX_ICP_DIST, prev_T_rel,
                o3d.pipelines.registration.TransformationEstimationPointToPlane())
            if icp_result.fitness >= ICP_FITNESS:
                prev_T_rel = icp_result.transformation.copy()
                if icp_pose is not None:
                    icp_pose = icp_pose @ np.linalg.inv(icp_result.transformation)
        prev_cloud = curr_cloud

        # ── Match against map ─────────────────────────────────────────────────
        matched_3d = []
        matched_2d = []

        if des is not None and len(map_des) >= MIN_MATCHES:
            for pair in matcher.knnMatch(des, map_des, k=2):
                if len(pair) < 2: continue
                m, n = pair
                if m.distance < RATIO_TEST * n.distance and m.distance < MAX_HAMMING:
                    matched_3d.append(map_pts[m.trainIdx])
                    matched_2d.append(kps[m.queryIdx].pt)

        # ── PnP ───────────────────────────────────────────────────────────────
        new_pose = pnp(np.array(matched_3d), np.array(matched_2d), icp_pose) \
                   if len(matched_3d) >= MIN_MATCHES else None

        if new_pose is not None:
            global_pose = new_pose
            icp_pose    = new_pose.copy()
            mode = "loc"
        elif icp_pose is not None:
            global_pose = icp_pose
            mode = "icp"
        else:
            mode = "lost"

        # ── Record trail ──────────────────────────────────────────────────────
        if global_pose is not None:
            trail_raw.append(global_pose[:3, 3].copy())

        # ── Open3D update ─────────────────────────────────────────────────────
        if len(trail_raw) >= 2:
            smoothed = smooth_trail(trail_raw, SMOOTH_WIN)
            trail_pcd.points = o3d.utility.Vector3dVector(smoothed)
            trail_pcd.paint_uniform_color([0.0, 0.8, 0.0])
            if not o3d_init:
                vis.add_geometry(trail_pcd)
                o3d_init = True
            else:
                vis.update_geometry(trail_pcd)

        vis.poll_events()
        vis.update_renderer()

        # ── OpenCV overlay ────────────────────────────────────────────────────
        dot = (0, 220, 0) if mode == "loc" else (0, 200, 200) if mode == "icp" else (0, 0, 220)
        for k in (kps or []):
            cv2.circle(dbg, (int(k.pt[0]), int(k.pt[1])), 2, dot, -1)

        pos_str = ""
        if global_pose is not None:
            p = global_pose[:3, 3]
            pos_str = f"  x:{p[0]:.2f}  y:{p[1]:.2f}  z:{p[2]:.2f}"

        cv2.putText(dbg,
                    f"mode:{mode}  matched:{len(matched_3d)}"
                    f"  trail:{len(trail_raw)}  f:{frame_idx}{pos_str}",
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
        cv2.imshow("localizer", dbg)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[stopped early]"); break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    vis.destroy_window()

# ── Save trail ────────────────────────────────────────────────────────────────
if trail_raw:
    np.save(TRAIL_OUT, np.array(trail_raw))
    print(f"\n[saved]  {TRAIL_OUT}  ({len(trail_raw)} poses)")
else:
    print("\n[warn]  No trail recorded — localisation never succeeded.")

print("[done]")
