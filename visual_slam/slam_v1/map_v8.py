"""
map_v8.py
---------
Replaces guided matching (Python loops) with LK optical flow (C++).

Architecture change vs v7
─────────────────────────
BEFORE (v7):  FAST+ORB → KpGrid (Python dict) → guided_match (Python loop over 4000 pts)
AFTER  (v8):  FAST     → calcOpticalFlowPyrLK (C++, ~2ms for 300 pts) → PnP

How tracking works:
  1. Detect FAST corners → lift 3D world position from depth → store as tracked set
  2. Every frame: LK moves each 2D point from prev frame to current frame (C++)
  3. Each point still knows its original 3D world position (fixed — world doesn't move)
  4. (2D_current, 3D_world) pairs → solvePnPRansac → camera pose
  5. When tracked count < MIN_TRACK: detect new FAST → lift 3D → add to set
  6. ORB kept only for fallback when LK loses all tracks

Python loops eliminated:
  ✗ KpGrid constructor    (was: Python dict built every frame)
  ✗ guided_match loop     (was: Python loop over 4000 map points)
  ✗ hamming() calls       (was: Python function called 4000x per frame)

Map building, 2D canvas, adaptive POS_ALPHA — same as v7.
Saves map_v8_pts.npy + map_v8_trajectory.npy on exit.

Controls → Q to quit
"""

import numpy as np
import cv2
import pyrealsense2 as rs
from collections import deque


# ── Settings ──────────────────────────────────────────────────────────────────
PLAYBACK_FILE = r"recordings\2026-04-12_15-11-52.bag"

FRAME_W     = 640
FRAME_H     = 480
FPS         = 30
MIN_DEPTH_M = 0.3
MAX_DEPTH_M = 4.0
VOXEL_SIZE  = 0.05
FRAME_SKIP  = 2
RENDER_EVERY = 3
CARVE_EVERY  = 10
FREE_MARGIN  = 0.10
TRAJ_SMOOTH_WIN = 15

# LK tracker settings
MAX_TRACK   = 300      # max points to track
MIN_TRACK   = 80       # refresh when below this
EXCL_RADIUS = 15       # px — don't detect near existing tracked points
LK_WIN      = (21, 21)
LK_LEVELS   = 3
LK_MAX_ERR  = 25.0

# PnP
MIN_MATCHES = 12
PNP_ITERS   = 100

# Keyframe (for ORB fallback only)
MAX_KF        = 4
KF_TRANS      = 0.15
KF_ROT_DEG    = 15.0
MIN_KF_FRAMES = 8

# Map display
CANVAS_W  = 900
CANVAS_H  = 900
MAP_SCALE = 100
OX        = CANVAS_W // 2
OZ        = CANVAS_H // 2


# ── LK parameters ─────────────────────────────────────────────────────────────
lk_params = dict(
    winSize  = LK_WIN,
    maxLevel = LK_LEVELS,
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
)


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

print(f"[D435]  fx={fx:.1f}  fy={fy:.1f}  cx={cx:.1f}  cy={cy:.1f}")

spatial  = rs.spatial_filter()
temporal = rs.temporal_filter()


# ── Detectors ─────────────────────────────────────────────────────────────────
fast    = cv2.FastFeatureDetector_create(threshold=10, nonmaxSuppression=True)
orb     = cv2.ORB_create(nfeatures=300)
matcher = cv2.BFMatcher(cv2.NORM_HAMMING)


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
    if len(pts) == 0: return
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
    if _map_n > 0: _draw_pts_on_canvas(_map_pts())


def render_2d(smooth_traj):
    frame = map_canvas.copy()
    if len(smooth_traj) >= 2:
        traj  = np.array(smooth_traj)
        tpx   = (OX + traj[:, 0] * MAP_SCALE).astype(np.int32)
        tpz   = (OZ - traj[:, 2] * MAP_SCALE).astype(np.int32)
        pts_l = np.stack([tpx, tpz], axis=1).reshape(-1, 1, 2)
        cv2.polylines(frame, [pts_l], False, (0, 0, 220), 2)
    if smooth_traj:
        px, pz = int(OX + smooth_traj[-1][0]*MAP_SCALE), int(OZ - smooth_traj[-1][2]*MAP_SCALE)
        if 0 <= px < CANVAS_W and 0 <= pz < CANVAS_H:
            cv2.circle(frame, (px, pz), 7, (0, 220, 0), -1)
    return frame


def map_add(pts_world):
    global _map_n
    if len(pts_world) == 0: return 0
    idx   = np.floor(pts_world / VOXEL_SIZE).astype(np.int32)
    new_i = [i for i in range(len(idx))
             if (key := (idx[i,0], idx[i,1], idx[i,2])) not in occupied
             and not occupied.add(key)]
    if not new_i: return 0
    new_pts = pts_world[new_i]
    end = min(_map_n + len(new_pts), _MAP_CAP)
    count = end - _map_n
    _map_buf[_map_n:end] = new_pts[:count]
    _map_n = end
    _draw_pts_on_canvas(new_pts[:count])
    return count


def carve_free_space(depth_raw, cam_pose):
    global _map_n
    if _map_n == 0: return 0
    pts     = _map_pts()
    w2c     = np.linalg.inv(cam_pose)
    pts_cam = (w2c[:3,:3] @ pts.T).T + w2c[:3,3]
    z = pts_cam[:,2]; u = pts_cam[:,0]/z*fx+cx; v = pts_cam[:,1]/z*fy+cy
    visible = (z>MIN_DEPTH_M)&(u>=0)&(u<FRAME_W-1)&(v>=0)&(v<FRAME_H-1)
    vis_idx = np.where(visible)[0]
    if len(vis_idx) == 0: return 0
    d_meas  = depth_raw[v[vis_idx].astype(int), u[vis_idx].astype(int)].astype(np.float32)*depth_scale
    in_free = (d_meas > MIN_DEPTH_M) & (z[vis_idx] < d_meas - FREE_MARGIN)
    rm_idx  = vis_idx[in_free]
    if len(rm_idx) == 0: return 0
    for k in np.floor(pts[rm_idx]/VOXEL_SIZE).astype(int):
        occupied.discard((k[0],k[1],k[2]))
    keep = np.ones(_map_n, dtype=bool); keep[rm_idx] = False
    kept = pts[keep]; _map_buf[:len(kept)] = kept; _map_n = len(kept)
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
    T[:3,:3] = R.T; T[:3,3] = -(R.T @ tvec.ravel())
    return T, len(inliers)


# ── LK tracker state ──────────────────────────────────────────────────────────
track_pts2d = np.zeros((0, 1, 2), dtype=np.float32)   # current 2D positions
track_pts3d = np.zeros((0, 3),    dtype=np.float64)   # fixed 3D world positions
prev_gray   = None


def _lift_3d(kps, depth_raw, cam_pose):
    """Convert FAST keypoints + depth → 3D world positions. Returns (pts2d, pts3d)."""
    R, t = cam_pose[:3,:3], cam_pose[:3,3]
    H, W = depth_raw.shape; WIN = 3
    p2, p3 = [], []
    for k in kps:
        u0, v0 = int(k.pt[0]), int(k.pt[1])
        patch = depth_raw[max(0,v0-WIN):min(H,v0+WIN+1),
                          max(0,u0-WIN):min(W,u0+WIN+1)].astype(np.float32)*depth_scale
        valid = patch[(patch>MIN_DEPTH_M)&(patch<MAX_DEPTH_M)]
        if len(valid) == 0: continue
        d  = float(np.median(valid))
        p3d = R @ np.array([(k.pt[0]-cx)/fx*d, (k.pt[1]-cy)/fy*d, d]) + t
        p2.append([[k.pt[0], k.pt[1]]])
        p3.append(p3d)
    if not p2:
        return np.zeros((0,1,2), np.float32), np.zeros((0,3), np.float64)
    return np.array(p2, np.float32), np.array(p3, np.float64)


def refresh_tracks(gray, depth_raw, cam_pose):
    """Detect new FAST features outside existing tracks and add them."""
    global track_pts2d, track_pts3d
    mask = np.full((FRAME_H, FRAME_W), 255, dtype=np.uint8)
    for pt in track_pts2d.reshape(-1, 2):
        cv2.circle(mask, (int(pt[0]), int(pt[1])), EXCL_RADIUS, 0, -1)
    kps = fast.detect(gray, mask)
    if not kps: return 0
    kps = sorted(kps, key=lambda k: k.response, reverse=True)[:MAX_TRACK]
    new2d, new3d = _lift_3d(kps, depth_raw, cam_pose)
    if len(new2d) == 0: return 0
    if len(track_pts2d) > 0:
        track_pts2d = np.vstack([track_pts2d, new2d])
        track_pts3d = np.vstack([track_pts3d, new3d])
    else:
        track_pts2d = new2d
        track_pts3d = new3d
    # Cap
    if len(track_pts2d) > MAX_TRACK:
        track_pts2d = track_pts2d[:MAX_TRACK]
        track_pts3d = track_pts3d[:MAX_TRACK]
    # Add to map
    map_add(new3d.astype(np.float32))
    return len(new2d)


# ── ORB fallback keyframes ─────────────────────────────────────────────────────
class KF:
    __slots__ = ('pose','des','pts3d')
    def __init__(self, pose, des, pts3d):
        self.pose=pose; self.des=des; self.pts3d=pts3d

kf_map = deque(maxlen=MAX_KF)
last_kf_pose    = np.eye(4)
frames_since_kf = 0


def make_kf(gray, depth_raw, cam_pose):
    kps, des = orb.detectAndCompute(gray, None)
    if des is None or len(kps) == 0: return
    _, pts3d = _lift_3d(kps, depth_raw, cam_pose)
    kf_map.append(KF(cam_pose.copy(), des, pts3d))


def orb_fallback(gray, depth_raw):
    """Try ORB match against stored keyframes when LK fails."""
    if not kf_map: return None, 0
    kps, des = orb.detectAndCompute(gray, None)
    if des is None or len(kps) == 0: return None, 0
    best_pose, best_n = None, 0
    for kf in reversed(kf_map):
        raw  = matcher.knnMatch(des, kf.des, k=2)
        good = [m for pair in raw if len(pair)==2
                for m,n in [(pair[0],pair[1])] if m.distance < 0.75*n.distance]
        if len(good) < MIN_MATCHES: continue
        p3 = np.array([kf.pts3d[m.trainIdx] for m in good
                       if not np.any(np.isnan(kf.pts3d[m.trainIdx]))], np.float64)
        p2 = np.array([kps[m.queryIdx].pt for m in good
                       if not np.any(np.isnan(kf.pts3d[m.trainIdx]))], np.float64)
        if len(p3) < MIN_MATCHES: continue
        pose, n = _pnp(p3, p2)
        if pose is not None and n > best_n:
            best_pose, best_n = pose, n
    return best_pose, best_n


def _rot_angle(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R)-1)/2,-1,1))))


def needs_kf(pose):
    if frames_since_kf < MIN_KF_FRAMES: return False
    if np.linalg.norm(pose[:3,3]-last_kf_pose[:3,3]) > KF_TRANS: return True
    if _rot_angle(pose[:3,:3].T @ last_kf_pose[:3,:3]) > KF_ROT_DEG: return True
    return False


# ── Main loop ─────────────────────────────────────────────────────────────────
print("\nMap v8 (LK flow) — move camera slowly.  Q to quit.\n")

global_pose  = np.eye(4)
velocity     = np.eye(4)
frame_idx    = 0
trajectory   = []
smooth_traj  = []
_traj_win    = deque(maxlen=TRAJ_SMOOTH_WIN)

try:
    while True:

        # ── Capture ───────────────────────────────────────────────────────────
        try:
            frames = align.process(pipeline.wait_for_frames(timeout_ms=5000))
        except RuntimeError:
            print("  [end of recording]"); break

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame: continue

        frame_idx += 1
        if frame_idx % FRAME_SKIP != 0: continue

        depth_frame = spatial.process(depth_frame)
        depth_frame = temporal.process(depth_frame)

        color_raw = np.asanyarray(color_frame.get_data())
        depth_raw = np.asanyarray(depth_frame.get_data())
        gray      = cv2.cvtColor(color_raw, cv2.COLOR_RGB2GRAY)

        # ── First frame: seed tracker ──────────────────────────────────────────
        if prev_gray is None:
            refresh_tracks(gray, depth_raw, global_pose)
            make_kf(gray, depth_raw, global_pose)
            last_kf_pose    = global_pose.copy()
            frames_since_kf = 0
            prev_gray       = gray
            print(f"  [seed]  {len(track_pts2d)} tracked pts  {_map_n} map pts")
            continue

        # ── LK tracking (C++ — fast) ───────────────────────────────────────────
        new_pose  = None
        n_inliers = 0

        if len(track_pts2d) >= MIN_MATCHES:
            new_pts, status, err = cv2.calcOpticalFlowPyrLK(
                prev_gray, gray, track_pts2d, None, **lk_params)

            good = status.ravel() == 1
            if err is not None:
                good &= err.ravel() < LK_MAX_ERR

            if good.sum() >= MIN_MATCHES:
                track_pts2d = new_pts[good]
                track_pts3d = track_pts3d[good]

                p2 = track_pts2d.reshape(-1, 2).astype(np.float64)
                p3 = track_pts3d.astype(np.float64)
                new_pose, n_inliers = _pnp(p3, p2)
            else:
                track_pts2d = new_pts[good] if good.any() else np.zeros((0,1,2), np.float32)
                track_pts3d = track_pts3d[good] if good.any() else np.zeros((0,3), np.float64)

        # ── ORB fallback if LK failed ──────────────────────────────────────────
        if new_pose is None:
            new_pose, n_inliers = orb_fallback(gray, depth_raw)
            if new_pose is not None:
                print(f"  [ORB fallback]  frame {frame_idx}  inliers={n_inliers}")
                # Reseed LK from current pose
                track_pts2d = np.zeros((0,1,2), np.float32)
                track_pts3d = np.zeros((0,3),   np.float64)
                refresh_tracks(gray, depth_raw, new_pose)

        # ── Update pose ───────────────────────────────────────────────────────
        if new_pose is not None:
            velocity  = np.linalg.inv(global_pose) @ new_pose
            vel_trans = np.linalg.norm(velocity[:3, 3])
            alpha     = float(np.clip(vel_trans / 0.05, 0.1, 0.9))
            new_pose[:3, 3] = alpha*new_pose[:3,3] + (1-alpha)*global_pose[:3,3]
            global_pose = new_pose
        else:
            global_pose = global_pose @ velocity
            velocity    = velocity * 0.5 + np.eye(4) * 0.5
            print(f"  [skip]  frame {frame_idx} — tracking lost")

        trajectory.append(global_pose[:3, 3].copy())
        _traj_win.append(global_pose[:3, 3].copy())
        smooth_traj.append(np.mean(_traj_win, axis=0))
        frames_since_kf += 1

        # ── Refresh tracker if sparse ──────────────────────────────────────────
        if len(track_pts2d) < MIN_TRACK:
            added = refresh_tracks(gray, depth_raw, global_pose)
            print(f"  [refresh]  +{added} pts  total={len(track_pts2d)}")

        # ── Keyframe for ORB fallback ──────────────────────────────────────────
        if needs_kf(global_pose):
            make_kf(gray, depth_raw, global_pose)
            last_kf_pose    = global_pose.copy()
            frames_since_kf = 0

        # ── Carve ─────────────────────────────────────────────────────────────
        if frame_idx % CARVE_EVERY == 0:
            removed = carve_free_space(depth_raw, global_pose)
            if removed: print(f"  [carve]  -{removed} pts")

        # ── 2D render ─────────────────────────────────────────────────────────
        if frame_idx % RENDER_EVERY == 0:
            cv2.imshow("Map v8 — 2D top-down", render_2d(smooth_traj))

        # ── Debug overlay ─────────────────────────────────────────────────────
        dbg = cv2.cvtColor(color_raw, cv2.COLOR_RGB2BGR)
        for pt in track_pts2d.reshape(-1, 2):
            cv2.circle(dbg, (int(pt[0]), int(pt[1])), 3, (0, 255, 0), -1)
        cv2.putText(dbg, f"tracked:{len(track_pts2d)}  map:{_map_n}  f:{frame_idx}",
                    (8,20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 1)
        cv2.putText(dbg, f"inliers:{n_inliers}",
                    (8,42), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,220,255), 1)
        cv2.imshow("map_v8 — camera", dbg)

        if cv2.waitKey(1) & 0xFF == ord('q'): break

        prev_gray = gray

finally:
    if _map_n > 0:
        np.save("map_v8_pts.npy", _map_pts().copy())
    if trajectory:
        np.save("map_v8_trajectory.npy", np.array(trajectory))
    print(f"\n[saved] map_v8_pts.npy         — {_map_n:,} points")
    print(f"[saved] map_v8_trajectory.npy  — {len(trajectory)} poses")
    pipeline.stop()
    cv2.destroyAllWindows()
    print("[done]")
