"""
map_builder.py
--------------
ICP-first offline map building:

  Every frame  →  ICP point-to-plane  →  accurate global pose
  At keyframe  →  ORB extracted (for descriptors only, NOT for pose)
                  existing map points observed  →  stored for BA
                  unmatched ORB + depth  →  new map points added
  End of bag   →  full bundle adjustment over all keyframes  →  map.npz

Because ICP uses all depth points the pose going into BA is already good.
BA then polishes small residual drift.

Run:  python map_builder.py
Then: python localizer.py
"""

import numpy as np
import cv2
import pyrealsense2 as rs
import open3d as o3d
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix


# ── Config ────────────────────────────────────────────────────────────────────
BAG_FILE  = r"..\recordings\2026-04-12_22-52-46.bag"
MAP_OUT   = "map.npz"

FRAME_W, FRAME_H = 640, 480
FRAME_SKIP  = 2
MIN_DEPTH_M = 0.3
MAX_DEPTH_M = 12.0
DEPTH_WIN   = 3

# ICP
PIXEL_STEP   = 4
VOXEL_DOWN   = 0.05
MAX_ICP_DIST = 0.3
ICP_FITNESS  = 0.20

# Keyframe decision
KF_TRANS      = 0.15    # m
KF_ROT_DEG    = 15.0    # degrees
MIN_KF_FRAMES = 8

# Map
VOXEL_SIZE  = 0.05
MAX_MAP_PTS = 50_000

# ORB (observation recording only — never used for pose estimation)
MAX_HAMMING = 80
RATIO_TEST  = 0.85

# BA
MIN_OBS_FOR_BA = 2
BA_MAX_NFEV    = 200

FX = FY = 384.327880859375
CX, CY  = 321.8272705078125, 239.01609802246094
K_MAT = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]], dtype=np.float64)
DIST  = np.zeros((4, 1), dtype=np.float64)


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


# ── ORB ───────────────────────────────────────────────────────────────────────
orb     = cv2.ORB_create(nfeatures=1500)
matcher = cv2.BFMatcher(cv2.NORM_HAMMING)


# ── Map state ─────────────────────────────────────────────────────────────────
_map_pts3d  = []    # (N,3) world positions — updated by BA
_map_des    = []    # (N,32) ORB descriptors — fixed
_map_obs    = []    # per-point: [[kf_idx, u, v], ...]
_map_voxels = set()
_keyframes  = []    # [{'idx': int, 'pose': 4x4}, ...]
_kf_count   = 0


# ── Helpers ───────────────────────────────────────────────────────────────────
def depth_at(depth_raw, u, v):
    u0, v0 = int(u), int(v)
    patch = depth_raw[max(0, v0-DEPTH_WIN):min(FRAME_H, v0+DEPTH_WIN+1),
                      max(0, u0-DEPTH_WIN):min(FRAME_W, u0+DEPTH_WIN+1)
                      ].astype(float) * depth_scale
    valid = patch[(patch > MIN_DEPTH_M) & (patch < MAX_DEPTH_M)]
    return float(np.median(valid)) if len(valid) else 0.0


def lift_to_world(u, v, z, pose):
    p_cam = np.array([(u - CX) * z / FX, (v - CY) * z / FY, z])
    return pose[:3, :3] @ p_cam + pose[:3, 3]


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
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    return pcd


def add_map_point(pos, des, kf_idx, u, v):
    key = tuple(np.floor(pos / VOXEL_SIZE).astype(int))
    if key in _map_voxels or len(_map_pts3d) >= MAX_MAP_PTS:
        return -1
    _map_voxels.add(key)
    idx = len(_map_pts3d)
    _map_pts3d.append(pos.copy())
    _map_des.append(des.copy())
    _map_obs.append([[kf_idx, float(u), float(v)]])
    return idx


def rot_deg(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def pose_to_6dof(pose):
    R_cw = pose[:3, :3].T
    rvec, _ = cv2.Rodrigues(R_cw)
    return np.concatenate([rvec.ravel(), -(R_cw @ pose[:3, 3])])


def dof6_to_pose(p):
    R_cw, _ = cv2.Rodrigues(p[:3])
    T = np.eye(4)
    T[:3, :3] = R_cw.T
    T[:3, 3]  = -(R_cw.T @ p[3:6])
    return T


# ── Bundle Adjustment ─────────────────────────────────────────────────────────
def run_full_ba():
    print("\n[BA]  Running full bundle adjustment …")
    n_kf = len(_keyframes)
    if n_kf < 2:
        print("[BA]  Not enough keyframes"); return

    kf_id_to_idx = {kf['idx']: i for i, kf in enumerate(_keyframes)}
    kf_id_set    = set(kf_id_to_idx)

    obs_list   = []
    active_set = set()
    for pt_idx, obs in enumerate(_map_obs):
        valid = [(kf_id_to_idx[o[0]], o[1], o[2])
                 for o in obs if o[0] in kf_id_set]
        if len(valid) >= MIN_OBS_FOR_BA:
            for wi, u, v in valid:
                obs_list.append((wi, pt_idx, u, v))
            active_set.add(pt_idx)

    active_pts  = sorted(active_set)
    pt_to_local = {pt: i for i, pt in enumerate(active_pts)}
    n_pts = len(active_pts)
    n_obs = len(obs_list)
    print(f"[BA]  keyframes:{n_kf}  points:{n_pts}  observations:{n_obs}")

    x0 = np.empty(n_kf * 6 + n_pts * 3)
    for i, kf in enumerate(_keyframes):
        x0[i*6:(i+1)*6] = pose_to_6dof(kf['pose'])
    for i, pt_idx in enumerate(active_pts):
        x0[n_kf*6 + i*3 : n_kf*6 + i*3+3] = _map_pts3d[pt_idx]

    sp = lil_matrix((n_obs * 2, n_kf * 6 + n_pts * 3), dtype=np.int8)
    for j, (wi, pt_idx, u, v) in enumerate(obs_list):
        loc = pt_to_local[pt_idx]
        sp[j*2:j*2+2, wi*6 : wi*6+6]                      = 1
        sp[j*2:j*2+2, n_kf*6+loc*3 : n_kf*6+loc*3+3]     = 1

    kf0_fix = x0[:6].copy()
    x_free  = x0[6:].copy()
    sp_free = sp[:, 6:].tocsr()

    def residuals(p):
        full = np.concatenate([kf0_fix, p])
        res  = np.empty(n_obs * 2)
        for j, (wi, pt_idx, u_obs, v_obs) in enumerate(obs_list):
            R_cw, _ = cv2.Rodrigues(full[wi*6:wi*6+3])
            tvec    = full[wi*6+3:wi*6+6]
            loc     = pt_to_local[pt_idx]
            X_w     = full[n_kf*6 + loc*3 : n_kf*6 + loc*3+3]
            X_c     = R_cw @ X_w + tvec
            if X_c[2] < 0.01:
                res[j*2] = res[j*2+1] = 50.0
                continue
            res[j*2]   = FX * X_c[0] / X_c[2] + CX - u_obs
            res[j*2+1] = FY * X_c[1] / X_c[2] + CY - v_obs
        return res

    print("[BA]  Optimising …")
    result = least_squares(residuals, x_free, jac_sparsity=sp_free,
                           method='trf', loss='huber',
                           max_nfev=BA_MAX_NFEV, verbose=2)

    full = np.concatenate([kf0_fix, result.x])
    for i in range(1, n_kf):
        _keyframes[i]['pose'] = dof6_to_pose(full[i*6:(i+1)*6])
    for i, pt_idx in enumerate(active_pts):
        _map_pts3d[pt_idx] = full[n_kf*6 + i*3 : n_kf*6 + i*3+3].copy()

    print(f"[BA]  Done  cost:{result.cost:.2f}  nfev:{result.nfev}")


# ── Main loop ─────────────────────────────────────────────────────────────────
global_pose     = np.eye(4)
last_kf_pose    = np.eye(4)
frames_since_kf = 0
frame_idx       = 0
seeded          = False
prev_cloud      = None
prev_T_rel      = np.eye(4)
icp_fitness     = 0.0

print("Building map … (Q to stop early)\n")

try:
    while True:
        try:
            frames = align.process(pipeline.wait_for_frames(timeout_ms=5000))
        except RuntimeError:
            print("[end of bag]"); break

        cf = frames.get_color_frame()
        df = frames.get_depth_frame()
        if not cf or not df:
            continue

        frame_idx += 1
        if frame_idx % FRAME_SKIP != 0:
            continue

        df = spatial.process(df)
        df = temporal.process(df)

        color_raw  = np.asanyarray(cf.get_data())
        depth_raw  = np.asanyarray(df.get_data())
        gray       = cv2.cvtColor(color_raw, cv2.COLOR_RGB2GRAY)
        dbg        = cv2.cvtColor(color_raw, cv2.COLOR_RGB2BGR)
        curr_cloud = depth_to_cloud(depth_raw)

        # ── Seed (first frame) ────────────────────────────────────────────────
        if not seeded:
            kf_idx   = _kf_count; _kf_count += 1
            kps, des = orb.detectAndCompute(gray, None)
            added    = 0
            if des is not None:
                for i, k in enumerate(kps):
                    z = depth_at(depth_raw, k.pt[0], k.pt[1])
                    if z > 0 and add_map_point(
                            lift_to_world(k.pt[0], k.pt[1], z, global_pose),
                            des[i], kf_idx, k.pt[0], k.pt[1]) >= 0:
                        added += 1
            _keyframes.append({'idx': kf_idx, 'pose': global_pose.copy()})
            last_kf_pose = global_pose.copy()
            prev_cloud   = curr_cloud
            seeded = True
            print(f"  [seed]  {added} pts  ({len(kps) if kps else 0} kps)")
            continue

        # ── ICP odometry — runs every frame, owns the pose ────────────────────
        icp_ok = False
        if prev_cloud is not None and curr_cloud is not None:
            icp_result = o3d.pipelines.registration.registration_icp(
                prev_cloud, curr_cloud, MAX_ICP_DIST, prev_T_rel,
                o3d.pipelines.registration.TransformationEstimationPointToPlane())
            icp_fitness = icp_result.fitness
            if icp_fitness >= ICP_FITNESS:
                prev_T_rel  = icp_result.transformation.copy()
                global_pose = global_pose @ np.linalg.inv(icp_result.transformation)
                icp_ok      = True
        prev_cloud = curr_cloud

        frames_since_kf += 1

        # ── Keyframe ──────────────────────────────────────────────────────────
        if icp_ok and frames_since_kf >= MIN_KF_FRAMES:
            dist = np.linalg.norm(global_pose[:3, 3] - last_kf_pose[:3, 3])
            rrot = rot_deg(global_pose[:3, :3].T @ last_kf_pose[:3, :3])
            if dist > KF_TRANS or rrot > KF_ROT_DEG:
                kf_idx   = _kf_count; _kf_count += 1
                kps, des = orb.detectAndCompute(gray, None)
                matched_query = set()
                new_obs = 0

                # Record observations for existing map points
                # (ORB matching here is for BA data only, not for pose)
                if des is not None and len(_map_des) >= 5:
                    map_des_arr = np.array(_map_des, dtype=np.uint8)
                    for pair in matcher.knnMatch(des, map_des_arr, k=2):
                        if len(pair) < 2: continue
                        m, n = pair
                        if (m.distance < RATIO_TEST * n.distance and
                                m.distance < MAX_HAMMING):
                            _map_obs[m.trainIdx].append([
                                kf_idx,
                                float(kps[m.queryIdx].pt[0]),
                                float(kps[m.queryIdx].pt[1])])
                            matched_query.add(m.queryIdx)
                            new_obs += 1

                # Add new map points from unmatched ORB features
                added = 0
                if des is not None:
                    for i, k in enumerate(kps):
                        if i in matched_query: continue
                        z = depth_at(depth_raw, k.pt[0], k.pt[1])
                        if z > 0 and add_map_point(
                                lift_to_world(k.pt[0], k.pt[1], z, global_pose),
                                des[i], kf_idx, k.pt[0], k.pt[1]) >= 0:
                            added += 1

                _keyframes.append({'idx': kf_idx, 'pose': global_pose.copy()})
                last_kf_pose    = global_pose.copy()
                frames_since_kf = 0
                print(f"  [KF {kf_idx:3d}]  f:{frame_idx:5d}"
                      f"  map:{len(_map_pts3d):5d}  +{added} new"
                      f"  obs:{new_obs}  dist:{dist:.2f}m"
                      f"  rot:{rrot:.1f}°  icp:{icp_fitness:.2f}")

        # ── Preview ───────────────────────────────────────────────────────────
        hud_col = (0, 220, 0) if icp_ok else (0, 80, 220)
        cv2.putText(dbg,
                    f"icp:{icp_fitness:.2f}  map:{len(_map_pts3d)}"
                    f"  kfs:{len(_keyframes)}  f:{frame_idx}",
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.50, hud_col, 1)
        cv2.putText(dbg,
                    f"x:{global_pose[0,3]:.2f}  y:{global_pose[1,3]:.2f}"
                    f"  z:{global_pose[2,3]:.2f}",
                    (8, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 220, 255), 1)
        cv2.imshow("map_builder", dbg)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[stopped early]"); break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()

print(f"\n[map]  {len(_map_pts3d)} points  {len(_keyframes)} keyframes")

# ── Full BA ───────────────────────────────────────────────────────────────────
run_full_ba()

# ── Save ──────────────────────────────────────────────────────────────────────
np.savez(MAP_OUT,
         pts3d = np.array(_map_pts3d, dtype=np.float64),
         des   = np.array(_map_des,   dtype=np.uint8),
         poses = np.array([kf['pose'] for kf in _keyframes], dtype=np.float64))

print(f"\n[saved]  {MAP_OUT}")
print(f"         pts3d : {len(_map_pts3d)}")
print(f"         des   : {len(_map_des)}")
print(f"         poses : {len(_keyframes)}")
print("\n[done]  Run localizer.py next.")
