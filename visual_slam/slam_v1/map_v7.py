"""
map_v7.py
---------
map_v6  +  3 speed optimisations for faster playback / Pi 5 readiness.

What changed vs v6
──────────────────
1. ORB_FEATURES   1000 → 500   (biggest single CPU saving)
2. PnP iterations  200 → 100   (halves RANSAC time, accuracy barely changes)
3. FRAME_SKIP = 2              (process every 2nd frame, halves total load)

Everything else identical to v6.
Saves map_v7_pts.npy + map_v7_trajectory.npy on exit.

Controls  →  Q to quit
"""

import numpy as np
import cv2
import pyrealsense2 as rs
from collections import deque


# ── Settings ──────────────────────────────────────────────────────────────────
PLAYBACK_FILE = r"recordings\2026-04-12_15-11-52.bag"

FRAME_W       = 640
FRAME_H       = 480
FPS           = 30
MIN_DEPTH_M   = 0.3
MAX_DEPTH_M   = 4.0
VOXEL_SIZE    = 0.05
ORB_FEATURES  = 500            # ← was 1000
PNP_ITERS     = 100            # ← was 200
FRAME_SKIP    = 2              # ← process every Nth frame
RENDER_EVERY  = 3
CARVE_EVERY   = 10
FREE_MARGIN   = 0.10
TRAJ_SMOOTH_WIN = 15

# Keyframe thresholds
MAX_KF        = 8
KF_TRANS      = 0.12
KF_ROT_DEG    = 12.0
MIN_KF_FRAMES = 5

# Guided matching
SEARCH_RADIUS = 30
MAX_HAMMING   = 60
MIN_MATCHES   = 12


# ── 2-D canvas settings ───────────────────────────────────────────────────────
CANVAS_W  = 900
CANVAS_H  = 900
MAP_SCALE = 100
OX        = CANVAS_W // 2
OZ        = CANVAS_H // 2


def _w2px(x, z):
    return int(OX + x * MAP_SCALE), int(OZ - z * MAP_SCALE)


# ── Hamming LUT ───────────────────────────────────────────────────────────────
_POPCOUNT = np.array([bin(i).count('1') for i in range(256)], dtype=np.int32)

def hamming(a, b):
    return int(_POPCOUNT[np.bitwise_xor(a, b)].sum())


# ── RealSense ─────────────────────────────────────────────────────────────────
pipeline = rs.pipeline()
cfg      = rs.config()

if PLAYBACK_FILE:
    cfg.enable_device_from_file(PLAYBACK_FILE, repeat_playback=False)
    profile = pipeline.start(cfg)
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


# ── FAST + ORB + BFMatcher ────────────────────────────────────────────────────
fast    = cv2.FastFeatureDetector_create(threshold=10, nonmaxSuppression=True)
orb     = cv2.ORB_create(nfeatures=ORB_FEATURES)
matcher = cv2.BFMatcher(cv2.NORM_HAMMING)


def detect_features(gray):
    kp = fast.detect(gray, None)
    if not kp:
        return [], None
    kp = sorted(kp, key=lambda k: k.response, reverse=True)[:ORB_FEATURES]
    kp, des = orb.compute(gray, kp)
    return kp, des


# ── Map storage ───────────────────────────────────────────────────────────────
_MAP_CAP      = 200_000
_map_buf      = np.zeros((_MAP_CAP, 3), dtype=np.float32)
_map_n        = 0
occupied: set = set()

def _map_pts():
    return _map_buf[:_map_n]

map_canvas = np.full((CANVAS_H, CANVAS_W, 3), 15, dtype=np.uint8)
cv2.line(map_canvas,   (OX, 0),  (OX, CANVAS_H), (45, 45, 45), 1)
cv2.line(map_canvas,   (0,  OZ), (CANVAS_W, OZ), (45, 45, 45), 1)
cv2.circle(map_canvas, (OX, OZ), 4, (80, 80, 80), -1)


def _draw_pts_on_canvas(pts):
    if len(pts) == 0:
        return
    px = (OX + pts[:, 0] * MAP_SCALE).astype(np.int32)
    pz = (OZ - pts[:, 2] * MAP_SCALE).astype(np.int32)
    v  = (px >= 0) & (px < CANVAS_W) & (pz >= 0) & (pz < CANVAS_H)
    map_canvas[pz[v], px[v]] = (140, 140, 140)


def _rebuild_canvas():
    global map_canvas
    map_canvas = np.full((CANVAS_H, CANVAS_W, 3), 15, dtype=np.uint8)
    cv2.line(map_canvas,   (OX, 0),  (OX, CANVAS_H), (45, 45, 45), 1)
    cv2.line(map_canvas,   (0,  OZ), (CANVAS_W, OZ), (45, 45, 45), 1)
    cv2.circle(map_canvas, (OX, OZ), 4, (80, 80, 80), -1)
    if _map_n > 0:
        _draw_pts_on_canvas(_map_pts())


def render_2d(smooth_traj):
    frame = map_canvas.copy()
    if len(smooth_traj) >= 2:
        traj  = np.array(smooth_traj)
        tpx   = (OX + traj[:, 0] * MAP_SCALE).astype(np.int32)
        tpz   = (OZ - traj[:, 2] * MAP_SCALE).astype(np.int32)
        pts_l = np.stack([tpx, tpz], axis=1).reshape(-1, 1, 2)
        cv2.polylines(frame, [pts_l], False, (0, 0, 220), 2)
    if smooth_traj:
        cx2, cz2 = _w2px(smooth_traj[-1][0], smooth_traj[-1][2])
        if 0 <= cx2 < CANVAS_W and 0 <= cz2 < CANVAS_H:
            cv2.circle(frame, (cx2, cz2), 7, (0, 220, 0), -1)
    return frame


# ── Keyframe ──────────────────────────────────────────────────────────────────
class Keyframe:
    __slots__ = ('id', 'pose', 'kp', 'des', 'pts3d')
    def __init__(self, id, pose, kp, des, pts3d):
        self.id=id; self.pose=pose; self.kp=kp; self.des=des; self.pts3d=pts3d

local_map: deque = deque(maxlen=MAX_KF)
_kf_id = 0


def create_keyframe(pose, gray, depth_raw, kp=None, des=None):
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
                        np.array([(u-cx)/fx*d, (v-cy)/fy*d, d]) + pose[:3, 3])
    kf = Keyframe(_kf_id, pose, kp, des, pts3d)
    _kf_id += 1
    return kf


# ── Spatial grid ──────────────────────────────────────────────────────────────
class KpGrid:
    def __init__(self, kp_list, des_array, cell=30):
        self.cell = cell; self.grid = {}
        for i, k in enumerate(kp_list):
            gx, gy = int(k.pt[0]/cell), int(k.pt[1]/cell)
            self.grid.setdefault((gx, gy), []).append((i, k.pt, des_array[i]))

    def query(self, u, v, radius):
        gx, gy = int(u/self.cell), int(v/self.cell)
        nc = int(radius/self.cell)+1; r2 = radius*radius
        return [item for dx in range(-nc, nc+1) for dy in range(-nc, nc+1)
                for item in self.grid.get((gx+dx, gy+dy), [])
                if (item[1][0]-u)**2+(item[1][1]-v)**2 <= r2]


# ── Map write / carve ─────────────────────────────────────────────────────────
def kp_to_world(kp_list, depth_raw, cam_pose):
    pts = []; R, t = cam_pose[:3, :3], cam_pose[:3, 3]
    H, W = depth_raw.shape; WIN = 3
    for k in kp_list:
        u0, v0 = int(k.pt[0]), int(k.pt[1])
        patch  = depth_raw[max(0,v0-WIN):min(H,v0+WIN+1),
                           max(0,u0-WIN):min(W,u0+WIN+1)].astype(np.float32)*depth_scale
        valid  = patch[(patch>MIN_DEPTH_M)&(patch<MAX_DEPTH_M)]
        if len(valid)==0: continue
        d = float(np.median(valid))
        pts.append(R @ np.array([(k.pt[0]-cx)/fx*d, (k.pt[1]-cy)/fy*d, d]) + t)
    return np.array(pts, np.float32) if pts else np.zeros((0,3), np.float32)


def map_add(pts_world):
    global _map_n
    if len(pts_world) == 0:
        return 0
    idx   = np.floor(pts_world / VOXEL_SIZE).astype(np.int32)
    new_i = [i for i in range(len(idx))
             if (key := (idx[i,0], idx[i,1], idx[i,2])) not in occupied
             and not occupied.add(key)]
    if not new_i:
        return 0
    new_pts = pts_world[new_i]
    end = min(_map_n + len(new_pts), _MAP_CAP)
    count = end - _map_n
    _map_buf[_map_n:end] = new_pts[:count]
    _map_n = end
    _draw_pts_on_canvas(new_pts[:count])
    return count


def carve_free_space(depth_raw, cam_pose):
    global _map_n
    if _map_n == 0:
        return 0
    pts     = _map_pts()
    w2c     = np.linalg.inv(cam_pose)
    pts_cam = (w2c[:3, :3] @ pts.T).T + w2c[:3, 3]
    z       = pts_cam[:, 2]
    u       = pts_cam[:, 0] / z * fx + cx
    v       = pts_cam[:, 1] / z * fy + cy
    visible = ((z > MIN_DEPTH_M) &
               (u >= 0) & (u < FRAME_W-1) &
               (v >= 0) & (v < FRAME_H-1))
    vis_idx = np.where(visible)[0]
    if len(vis_idx) == 0:
        return 0
    d_meas  = depth_raw[v[vis_idx].astype(int), u[vis_idx].astype(int)].astype(np.float32)*depth_scale
    in_free = (d_meas > MIN_DEPTH_M) & (z[vis_idx] < d_meas - FREE_MARGIN)
    rm_idx  = vis_idx[in_free]
    if len(rm_idx) == 0:
        return 0
    rm_keys = np.floor(pts[rm_idx] / VOXEL_SIZE).astype(int)
    for k in rm_keys:
        occupied.discard((k[0], k[1], k[2]))
    keep = np.ones(_map_n, dtype=bool); keep[rm_idx] = False
    kept = pts[keep]
    _map_buf[:len(kept)] = kept
    _map_n = len(kept)
    _rebuild_canvas()
    return len(rm_idx)


# ── PnP ───────────────────────────────────────────────────────────────────────
def _pnp(pts3d, pts2d):
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        pts3d, pts2d, K, dist,
        iterationsCount=PNP_ITERS, reprojectionError=4.0, confidence=0.99)
    if not ok or inliers is None or len(inliers) < MIN_MATCHES:
        return None, 0
    R, _ = cv2.Rodrigues(rvec); T = np.eye(4)
    T[:3, :3] = R.T; T[:3, 3] = -(R.T @ tvec.ravel())
    return T, len(inliers)


def project_local_map(predicted_pose):
    if not local_map:
        return np.zeros((0,3)), np.zeros((0,32), dtype=np.uint8), np.zeros((0,2))
    w2c = np.linalg.inv(predicted_pose); R, t = w2c[:3,:3], w2c[:3,3]
    ap, ad, au = [], [], []
    for kf in local_map:
        valid = ~np.any(np.isnan(kf.pts3d), axis=1)
        if not valid.any(): continue
        pts = kf.pts3d[valid]; des = kf.des[valid]
        pc  = (R @ pts.T).T + t; z = pc[:, 2]; front = z > MIN_DEPTH_M
        if not front.any(): continue
        u2 = pc[front,0]/z[front]*fx+cx; v2 = pc[front,1]/z[front]*fy+cy
        ib = (u2>=0)&(u2<FRAME_W)&(v2>=0)&(v2<FRAME_H)
        ap.append(pts[front][ib]); ad.append(des[front][ib])
        au.append(np.stack([u2[ib], v2[ib]], axis=1))
    if not ap:
        return np.zeros((0,3)), np.zeros((0,32), dtype=np.uint8), np.zeros((0,2))
    return np.vstack(ap), np.vstack(ad), np.vstack(au)


def guided_match(curr_kp, curr_des, map_pts3d, map_des, map_uv):
    if curr_des is None or len(map_pts3d)==0:
        return np.zeros((0,3), np.float64), np.zeros((0,2), np.float64)
    grid = KpGrid(curr_kp, curr_des); p3, p2, used = [], [], set()
    for pt3d, des_map, (up, vp) in zip(map_pts3d, map_des, map_uv):
        cands = grid.query(up, vp, SEARCH_RADIUS)
        bd, bi, buv = MAX_HAMMING, -1, None
        for idx, pt2d, des_c in cands:
            if idx in used: continue
            d = hamming(des_map, des_c)
            if d < bd: bd=d; bi=idx; buv=pt2d
        if bi >= 0: used.add(bi); p3.append(pt3d); p2.append(buv)
    if not p3:
        return np.zeros((0,3), np.float64), np.zeros((0,2), np.float64)
    return np.array(p3, np.float64), np.array(p2, np.float64)


def fallback_pose(curr_kp, curr_des):
    if not local_map or curr_des is None: return None, 0
    lkf = local_map[-1]
    if lkf.des is None: return None, 0
    raw  = matcher.knnMatch(curr_des, lkf.des, k=2)
    good = [m for pair in raw if len(pair)==2
            for m, n in [(pair[0], pair[1])] if m.distance < 0.75*n.distance]
    if len(good) < MIN_MATCHES: return None, 0
    p3, p2 = [], []
    for m in good:
        pt = lkf.pts3d[m.trainIdx]
        if np.any(np.isnan(pt)): continue
        p3.append(pt); p2.append(curr_kp[m.queryIdx].pt)
    if len(p3) < MIN_MATCHES: return None, 0
    return _pnp(np.array(p3, np.float64), np.array(p2, np.float64))


def _rot_angle(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R)-1)/2, -1, 1))))


def needs_keyframe(curr_pose, last_kf_pose, n_matches, frames_since_kf):
    if frames_since_kf < MIN_KF_FRAMES: return False
    if np.linalg.norm(curr_pose[:3,3]-last_kf_pose[:3,3]) > KF_TRANS: return True
    if _rot_angle(curr_pose[:3,:3].T @ last_kf_pose[:3,:3]) > KF_ROT_DEG: return True
    if n_matches < MIN_MATCHES*2: return True
    return False


# ── Main loop ─────────────────────────────────────────────────────────────────
print("\nMap v7 — move camera slowly.  Q to quit.\n")

global_pose     = np.eye(4)
velocity        = np.eye(4)
last_kf_pose    = np.eye(4)
frames_since_kf = 0
frame_idx       = 0
trajectory      = []
smooth_traj     = []
_traj_window    = deque(maxlen=TRAJ_SMOOTH_WIN)

try:
    while True:

        # ── Capture ───────────────────────────────────────────────────────────
        try:
            frames = align.process(pipeline.wait_for_frames(timeout_ms=5000))
        except RuntimeError:
            print("  [end of recording]")
            break
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            continue

        # ── Frame skip ────────────────────────────────────────────────────────
        frame_idx += 1
        if frame_idx % FRAME_SKIP != 0:
            continue

        depth_frame = spatial.process(depth_frame)
        depth_frame = temporal.process(depth_frame)

        color_raw = np.asanyarray(color_frame.get_data())
        depth_raw = np.asanyarray(depth_frame.get_data())
        gray      = cv2.cvtColor(color_raw, cv2.COLOR_RGB2GRAY)
        kp, des   = detect_features(gray)

        # ── Seed ──────────────────────────────────────────────────────────────
        if not local_map:
            kf = create_keyframe(global_pose, gray, depth_raw, kp, des)
            if kf:
                local_map.append(kf)
                last_kf_pose    = global_pose.copy()
                frames_since_kf = 0
                map_add(kp_to_world(kp, depth_raw, global_pose))
                print(f"  [KF 0] seeded — {_map_n:,} pts")
            continue

        # ── Predict ───────────────────────────────────────────────────────────
        predicted_pose = global_pose @ velocity

        # ── Guided match ──────────────────────────────────────────────────────
        map_pts3d, map_des, map_uv = project_local_map(predicted_pose)
        new_pose  = None
        n_inliers = 0

        if len(map_pts3d) >= MIN_MATCHES:
            pts3d, pts2d = guided_match(kp, des, map_pts3d, map_des, map_uv)
            if len(pts3d) >= MIN_MATCHES:
                new_pose, n_inliers = _pnp(pts3d, pts2d)

        # ── Fallback ──────────────────────────────────────────────────────────
        if new_pose is None:
            new_pose, n_inliers = fallback_pose(kp, des)
            if new_pose is not None:
                print(f"  [fallback]  frame {frame_idx}  inliers={n_inliers}")

        # ── Update pose ───────────────────────────────────────────────────────
        if new_pose is not None:
            velocity  = np.linalg.inv(global_pose) @ new_pose
            vel_trans = np.linalg.norm(velocity[:3, 3])
            alpha     = float(np.clip(vel_trans / 0.05, 0.1, 0.9))
            new_pose[:3, 3] = alpha * new_pose[:3, 3] + (1-alpha) * global_pose[:3, 3]
            global_pose = new_pose
        else:
            global_pose = predicted_pose
            velocity    = velocity * 0.5 + np.eye(4) * 0.5
            print(f"  [skip]  frame {frame_idx} — tracking lost")

        trajectory.append(global_pose[:3, 3].copy())
        _traj_window.append(global_pose[:3, 3].copy())
        smooth_traj.append(np.mean(_traj_window, axis=0))
        frames_since_kf += 1

        # ── Keyframe ──────────────────────────────────────────────────────────
        if needs_keyframe(global_pose, last_kf_pose, n_inliers, frames_since_kf):
            kf = create_keyframe(global_pose, gray, depth_raw, kp, des)
            if kf:
                local_map.append(kf)
                last_kf_pose    = global_pose.copy()
                frames_since_kf = 0
                print(f"  [KF {kf.id}]  frame {frame_idx}  "
                      f"local_map={len(local_map)}  inliers={n_inliers}")

        # ── Carve free space ──────────────────────────────────────────────────
        if frame_idx % CARVE_EVERY == 0:
            removed = carve_free_space(depth_raw, global_pose)
            if removed:
                print(f"  [carve]  -{removed} pts")

        # ── Add map points ────────────────────────────────────────────────────
        added = map_add(kp_to_world(kp, depth_raw, global_pose))

        # ── 2-D render ────────────────────────────────────────────────────────
        if frame_idx % RENDER_EVERY == 0:
            cv2.imshow("Map v7 — 2D top-down", render_2d(smooth_traj))

        # ── Debug overlay ─────────────────────────────────────────────────────
        dbg = cv2.cvtColor(color_raw, cv2.COLOR_RGB2BGR)
        for k in kp:
            cv2.circle(dbg, (int(k.pt[0]), int(k.pt[1])), 3, (0,255,0), -1)
        cv2.putText(dbg, f"feats:{len(kp)}  KFs:{len(local_map)}  "
                         f"map:{_map_n}  f:{frame_idx}",
                    (8,20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 1)
        cv2.putText(dbg, f"inliers:{n_inliers}  added:{added}",
                    (8,42), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,220,255), 1)
        cv2.imshow("map_v7 — camera", dbg)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    if _map_n > 0:
        np.save("map_v7_pts.npy", _map_pts().copy())
    if trajectory:
        np.save("map_v7_trajectory.npy", np.array(trajectory))
    print(f"\n[saved] map_v7_pts.npy          — {_map_n:,} points")
    print(f"[saved] map_v7_trajectory.npy   — {len(trajectory)} poses")
    pipeline.stop()
    cv2.destroyAllWindows()
    print("[done]")
