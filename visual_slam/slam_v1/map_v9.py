"""
map_v9.py
---------
map_v8  +  multi-threading  +  loop closure  +  pose graph correction.

Three threads
─────────────
  Main     — RealSense capture + LK tracking + PnP + display
  Mapping  — ORB compute + map_add + free-space carving  (background)
  LC       — loop closure detection against global KF database  (background)

Loop closure
────────────
  1. Every new keyframe: mapping thread computes ORB → sends GlobalKF to LC thread
  2. LC thread compares new KF against all KFs older than MIN_LC_SEP
  3. Verify candidate with 3D-2D PnP  (old KF 3D pts  ×  new KF 2D pts)
  4. If inliers ≥ LC_MIN_INLIERS → confirmed → put correction in corr_queue

Pose correction  (linear distribution)
───────────────────────────────────────
  After LC matches current KF (traj index N) to old KF (traj index K):
    drift = lc_corrected_pos − tracked_pos
    trajectory[i] += (i−K)/(N−K) × drift   for i in [K..N]
  Distributes drift smoothly — no sudden jump in the trail.

Saves map_v9_pts.npy + map_v9_trajectory.npy on exit.
Controls → Q to quit
"""

import numpy as np
import cv2
import pyrealsense2 as rs
import threading
import queue
from collections import deque


# ── Settings ──────────────────────────────────────────────────────────────────
PLAYBACK_FILE   = r"recordings\2026-04-12_21-34-13.bag"

FRAME_W         = 640
FRAME_H         = 480
FPS             = 30
MIN_DEPTH_M     = 0.3
MAX_DEPTH_M     = 4.0
VOXEL_SIZE      = 0.05
FRAME_SKIP      = 2
RENDER_EVERY    = 3
CARVE_EVERY     = 10
FREE_MARGIN     = 0.10
TRAJ_SMOOTH_WIN = 15

# LK tracker
MAX_TRACK   = 300
MIN_TRACK   = 80
EXCL_RADIUS = 15
LK_WIN      = (21, 21)
LK_LEVELS   = 3
LK_MAX_ERR  = 25.0

# PnP
MIN_MATCHES = 12
PNP_ITERS   = 100

# Keyframe (triggers mapping + LC)
KF_TRANS        = 0.15
KF_ROT_DEG      = 15.0
MIN_KF_FRAMES   = 8

# Loop closure
MIN_LC_SEP      = 10     # skip this many recent KFs when searching
MAX_LC_CHECK    = 150    # how many old KFs to check each time
LC_MIN_MATCHES  = 8      # descriptor matches needed to try PnP
LC_MIN_INLIERS  = 8      # PnP inliers needed to confirm loop

# 2D canvas
CANVAS_W  = 900
CANVAS_H  = 900
MAP_SCALE = 100
OX        = CANVAS_W // 2
OZ        = CANVAS_H // 2


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
fx, fy   = intr.fx, intr.fy
cx, cy   = intr.ppx, intr.ppy
K        = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], dtype=np.float64)
dist     = np.zeros((4,1), dtype=np.float64)

print(f"[D435]  fx={fx:.1f}  fy={fy:.1f}  cx={cx:.1f}  cy={cy:.1f}")

spatial  = rs.spatial_filter()
temporal = rs.temporal_filter()


# ── Detectors (main thread) ───────────────────────────────────────────────────
fast = cv2.FastFeatureDetector_create(threshold=10, nonmaxSuppression=True)
# NOTE: each thread gets its own ORB + BFMatcher — OpenCV objects are not thread-safe


# ── Map storage ───────────────────────────────────────────────────────────────
_MAP_CAP  = 200_000
_map_buf  = np.zeros((_MAP_CAP, 3), dtype=np.float32)
_map_n    = 0
occupied  = set()
map_lock  = threading.Lock()

def _map_pts():
    return _map_buf[:_map_n]

map_canvas = np.full((CANVAS_H, CANVAS_W, 3), 15, dtype=np.uint8)
cv2.line(map_canvas,   (OX, 0),  (OX, CANVAS_H), (45,45,45), 1)
cv2.line(map_canvas,   (0,  OZ), (CANVAS_W, OZ), (45,45,45), 1)
cv2.circle(map_canvas, (OX, OZ), 4, (80,80,80), -1)


def _draw_pts_on_canvas(pts):
    if len(pts) == 0: return
    px = (OX + pts[:,0] * MAP_SCALE).astype(np.int32)
    pz = (OZ - pts[:,2] * MAP_SCALE).astype(np.int32)
    v  = (px>=0)&(px<CANVAS_W)&(pz>=0)&(pz<CANVAS_H)
    map_canvas[pz[v], px[v]] = (140,140,140)


def _rebuild_canvas():
    global map_canvas
    map_canvas = np.full((CANVAS_H, CANVAS_W, 3), 15, dtype=np.uint8)
    cv2.line(map_canvas,   (OX, 0),  (OX, CANVAS_H), (45,45,45), 1)
    cv2.line(map_canvas,   (0,  OZ), (CANVAS_W, OZ), (45,45,45), 1)
    cv2.circle(map_canvas, (OX, OZ), 4, (80,80,80), -1)
    if _map_n > 0: _draw_pts_on_canvas(_map_pts())


def map_add(pts_world):
    global _map_n
    if len(pts_world) == 0: return 0
    idx   = np.floor(pts_world / VOXEL_SIZE).astype(np.int32)
    new_i = [i for i in range(len(idx))
             if (key := (idx[i,0],idx[i,1],idx[i,2])) not in occupied
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
    vis = (z>MIN_DEPTH_M)&(u>=0)&(u<FRAME_W-1)&(v>=0)&(v<FRAME_H-1)
    vi  = np.where(vis)[0]
    if len(vi)==0: return 0
    dm  = depth_raw[v[vi].astype(int), u[vi].astype(int)].astype(np.float32)*depth_scale
    rm  = vi[(dm>MIN_DEPTH_M)&(z[vi]<dm-FREE_MARGIN)]
    if len(rm)==0: return 0
    for k in np.floor(pts[rm]/VOXEL_SIZE).astype(int):
        occupied.discard((k[0],k[1],k[2]))
    keep = np.ones(_map_n, dtype=bool); keep[rm] = False
    kept = pts[keep]; _map_buf[:len(kept)] = kept; _map_n = len(kept)
    _rebuild_canvas()
    return len(rm)


# ── Shared trajectory ─────────────────────────────────────────────────────────
trajectory   = []       # raw positions  (main thread writes, LC corrects via corr_queue)
smooth_traj  = []       # display only
traj_lock    = threading.Lock()


def render_2d():
    with map_lock:
        frame = map_canvas.copy()
    with traj_lock:
        st = list(smooth_traj)
    if len(st) >= 2:
        arr  = np.array(st)
        tpx  = (OX + arr[:,0]*MAP_SCALE).astype(np.int32)
        tpz  = (OZ - arr[:,2]*MAP_SCALE).astype(np.int32)
        cv2.polylines(frame, [np.stack([tpx,tpz],1).reshape(-1,1,2)], False, (0,0,220), 2)
    if st:
        px = int(OX + st[-1][0]*MAP_SCALE); pz = int(OZ - st[-1][2]*MAP_SCALE)
        if 0<=px<CANVAS_W and 0<=pz<CANVAS_H:
            cv2.circle(frame, (px,pz), 7, (0,220,0), -1)
    return frame


# ── PnP helper ────────────────────────────────────────────────────────────────
def _pnp(pts3d, pts2d, iters=PNP_ITERS):
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        pts3d, pts2d, K, dist,
        iterationsCount=iters, reprojectionError=4.0, confidence=0.99)
    if not ok or inliers is None or len(inliers) < MIN_MATCHES:
        return None, 0
    R,_ = cv2.Rodrigues(rvec); T = np.eye(4)
    T[:3,:3]=R.T; T[:3,3]=-(R.T@tvec.ravel())
    return T, len(inliers)


# ── Depth lift helper (shared) ────────────────────────────────────────────────
def _lift_3d(kps, depth_raw, cam_pose):
    """FAST/ORB keypoints → (pts2d, pts3d_world).  Returns numpy arrays."""
    R, t = cam_pose[:3,:3], cam_pose[:3,3]
    H, W = depth_raw.shape; WIN = 3
    p2, p3 = [], []
    for k in kps:
        u0,v0 = int(k.pt[0]), int(k.pt[1])
        patch = depth_raw[max(0,v0-WIN):min(H,v0+WIN+1),
                          max(0,u0-WIN):min(W,u0+WIN+1)].astype(np.float32)*depth_scale
        valid = patch[(patch>MIN_DEPTH_M)&(patch<MAX_DEPTH_M)]
        if len(valid)==0: continue
        d   = float(np.median(valid))
        p3d = R @ np.array([(k.pt[0]-cx)/fx*d,(k.pt[1]-cy)/fy*d,d]) + t
        p2.append([[k.pt[0],k.pt[1]]])
        p3.append(p3d)
    if not p2:
        return np.zeros((0,1,2),np.float32), np.zeros((0,3),np.float64)
    return np.array(p2,np.float32), np.array(p3,np.float64)


# ── LK tracker state (main thread only — no lock needed) ─────────────────────
track_pts2d = np.zeros((0,1,2), np.float32)
track_pts3d = np.zeros((0,3),   np.float64)
prev_gray   = None

lk_params = dict(winSize=LK_WIN, maxLevel=LK_LEVELS,
                 criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,20,0.03))


def refresh_tracks(gray, depth_raw, cam_pose):
    global track_pts2d, track_pts3d
    mask = np.full((FRAME_H,FRAME_W), 255, dtype=np.uint8)
    for pt in track_pts2d.reshape(-1,2):
        cv2.circle(mask,(int(pt[0]),int(pt[1])),EXCL_RADIUS,0,-1)
    kps = fast.detect(gray, mask)
    if not kps: return 0
    kps = sorted(kps, key=lambda k:k.response, reverse=True)[:MAX_TRACK]
    new2d, new3d = _lift_3d(kps, depth_raw, cam_pose)
    if len(new2d)==0: return 0
    track_pts2d = np.vstack([track_pts2d,new2d]) if len(track_pts2d) else new2d
    track_pts3d = np.vstack([track_pts3d,new3d]) if len(track_pts3d) else new3d
    if len(track_pts2d)>MAX_TRACK:
        track_pts2d=track_pts2d[:MAX_TRACK]; track_pts3d=track_pts3d[:MAX_TRACK]
    with map_lock:
        map_add(new3d.astype(np.float32))
    return len(new2d)


# ── GlobalKF (shared between mapping + LC threads) ────────────────────────────
class GlobalKF:
    __slots__ = ('id','traj_idx','pose','des','pts3d','kpts_uv')
    def __init__(self, id, traj_idx, pose, des, pts3d, kpts_uv):
        self.id=id; self.traj_idx=traj_idx; self.pose=pose
        self.des=des; self.pts3d=pts3d; self.kpts_uv=kpts_uv

# Recent KFs stored here for ORB fallback in main thread
recent_kfs      = deque(maxlen=4)
recent_kfs_lock = threading.Lock()


# ── Inter-thread queues ───────────────────────────────────────────────────────
map_queue  = queue.Queue(maxsize=8)   # main  → mapping
lc_queue   = queue.Queue(maxsize=40)  # mapping → LC
corr_queue = queue.Queue(maxsize=1)   # LC → main  (only latest correction matters)

running = threading.Event()
running.set()


# ── Mapping thread ────────────────────────────────────────────────────────────
def mapping_worker():
    orb_m    = cv2.ORB_create(nfeatures=300)
    kf_count = 0
    while running.is_set():
        try:
            gray, depth_raw, pose, traj_idx = map_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        # ORB descriptors + 2D positions
        kps, des = orb_m.detectAndCompute(gray, None)
        if des is None or len(kps) == 0:
            continue

        # Lift 3D for each ORB keypoint
        _, pts3d_orb = _lift_3d(kps, depth_raw, pose)
        kpts_uv = np.array([[k.pt[0],k.pt[1]] for k in kps], np.float32)

        # Trim to only keypoints that got valid depth
        valid = ~np.any(np.isnan(pts3d_orb), axis=1) if len(pts3d_orb) else np.array([],bool)
        # pts3d_orb may be shorter than kps if _lift_3d skipped some
        # rebuild matching arrays
        valid_kps = []
        valid_des = []
        valid_uvs = []
        valid_3d  = []
        depth_kp_idx = 0
        for i, k in enumerate(kps):
            # check if this kp got a 3D point
            if depth_kp_idx < len(pts3d_orb):
                # _lift_3d only appends when depth is valid, so we need to track count
                pass
        # Simpler: recompute directly
        pts3d_list = []
        kpts_uv_list = []
        des_list = []
        R, t = pose[:3,:3], pose[:3,3]
        H, W = depth_raw.shape; WIN = 3
        for i, k in enumerate(kps):
            u0,v0 = int(k.pt[0]), int(k.pt[1])
            patch = depth_raw[max(0,v0-WIN):min(H,v0+WIN+1),
                              max(0,u0-WIN):min(W,u0+WIN+1)].astype(np.float32)*depth_scale
            vld = patch[(patch>MIN_DEPTH_M)&(patch<MAX_DEPTH_M)]
            if len(vld)==0: continue
            d   = float(np.median(vld))
            p3d = R@np.array([(k.pt[0]-cx)/fx*d,(k.pt[1]-cy)/fy*d,d])+t
            pts3d_list.append(p3d)
            kpts_uv_list.append([k.pt[0],k.pt[1]])
            des_list.append(des[i])

        if len(pts3d_list) < MIN_MATCHES:
            continue

        pts3d_arr = np.array(pts3d_list, np.float64)
        kpts_uv_arr = np.array(kpts_uv_list, np.float32)
        des_arr   = np.array(des_list, np.uint8)

        gkf = GlobalKF(kf_count, traj_idx, pose.copy(),
                       des_arr, pts3d_arr, kpts_uv_arr)
        kf_count += 1

        # Store for ORB fallback
        with recent_kfs_lock:
            recent_kfs.append(gkf)

        # Send to LC thread
        try:
            lc_queue.put_nowait(gkf)
        except queue.Full:
            pass

        # Add map points + carve
        with map_lock:
            map_add(pts3d_arr.astype(np.float32))
            if kf_count % CARVE_EVERY == 0:
                carve_free_space(depth_raw, pose)


# ── Loop Closure thread ───────────────────────────────────────────────────────
def lc_worker():
    kf_db    = []
    matcher_lc = cv2.BFMatcher(cv2.NORM_HAMMING)

    while running.is_set():
        try:
            kf = lc_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        kf_db.append(kf)

        # Need enough separation before we can find a loop
        if len(kf_db) < MIN_LC_SEP + 2:
            continue

        # Search candidates: skip recent MIN_LC_SEP KFs
        candidates = kf_db[:-MIN_LC_SEP]
        # Only check the most recent window of candidates (performance)
        candidates = candidates[-MAX_LC_CHECK:]

        best_pose   = None
        best_n      = 0
        best_old_kf = None
        best_matches = 0   # debug: track best match count even if PnP fails

        for old_kf in candidates:
            # Descriptor matching: old KF (train) vs new KF (query)
            raw = matcher_lc.knnMatch(kf.des, old_kf.des, k=2)
            good = [m for pair in raw if len(pair)==2
                    for m,n in [(pair[0],pair[1])]
                    if m.distance < 0.75*n.distance]
            if len(good) > best_matches:
                best_matches = len(good)
            if len(good) < LC_MIN_MATCHES:
                continue

            # 3D-2D PnP:
            #   3D = old KF world positions  (train index)
            #   2D = new KF pixel positions  (query index)
            p3 = np.array([old_kf.pts3d[m.trainIdx] for m in good], np.float64)
            p2 = np.array([kf.kpts_uv[m.queryIdx]   for m in good], np.float64)

            pose_lc, n_inl = _pnp(p3, p2, iters=150)
            if pose_lc is None or n_inl < LC_MIN_INLIERS:
                continue

            if n_inl > best_n:
                best_n      = n_inl
                best_pose   = pose_lc
                best_old_kf = old_kf

        if best_pose is None:
            if best_matches > 0:
                print(f"  [LC]  KF {kf.id}  best_matches={best_matches}  (need {LC_MIN_MATCHES} + PnP {LC_MIN_INLIERS})")
            continue

        # Confirmed loop closure
        drift = best_pose[:3,3] - kf.pose[:3,3]
        drift_mag = np.linalg.norm(drift)

        # Only correct if drift is meaningful (> 5cm) and not crazy large (> 2m)
        if drift_mag < 0.03 or drift_mag > 5.0:
            continue

        correction = {
            'drift_pos'      : drift,
            'traj_start_idx' : best_old_kf.traj_idx,
            'traj_end_idx'   : kf.traj_idx,
        }

        # Replace old correction if pending (only keep latest)
        while not corr_queue.empty():
            try: corr_queue.get_nowait()
            except queue.Empty: break
        try:
            corr_queue.put_nowait(correction)
            print(f"\n  [LOOP CLOSURE]  KF {best_old_kf.id} → KF {kf.id}"
                  f"  drift={drift_mag:.3f}m  inliers={best_n}")
        except queue.Full:
            pass


# ── ORB fallback (main thread, uses recent_kfs) ───────────────────────────────
_orb_fb  = cv2.ORB_create(nfeatures=300)
_mat_fb  = cv2.BFMatcher(cv2.NORM_HAMMING)

def orb_fallback(gray, depth_raw):
    kps, des = _orb_fb.detectAndCompute(gray, None)
    if des is None: return None, 0
    with recent_kfs_lock:
        kfs = list(recent_kfs)
    best_pose, best_n = None, 0
    for kf in reversed(kfs):
        raw  = _mat_fb.knnMatch(des, kf.des, k=2)
        good = [m for pair in raw if len(pair)==2
                for m,n in [(pair[0],pair[1])] if m.distance<0.75*n.distance]
        if len(good)<MIN_MATCHES: continue
        p3 = np.array([kf.pts3d[m.trainIdx] for m in good], np.float64)
        p2 = np.array([kps[m.queryIdx].pt   for m in good], np.float64)
        pose,n = _pnp(p3, p2)
        if pose is not None and n>best_n:
            best_pose,best_n = pose,n
    return best_pose, best_n


def _rot_angle(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R)-1)/2,-1,1))))

last_kf_pose    = np.eye(4)
frames_since_kf = 0
_kf_traj_idx    = 0


def needs_kf(pose):
    global last_kf_pose
    if frames_since_kf < MIN_KF_FRAMES: return False
    if np.linalg.norm(pose[:3,3]-last_kf_pose[:3,3]) > KF_TRANS: return True
    if _rot_angle(pose[:3,:3].T@last_kf_pose[:3,:3]) > KF_ROT_DEG: return True
    return False


# ── Start background threads ──────────────────────────────────────────────────
t_map = threading.Thread(target=mapping_worker, daemon=True, name="Mapping")
t_lc  = threading.Thread(target=lc_worker,      daemon=True, name="LC")
t_map.start()
t_lc.start()
print("[threads]  Mapping + LoopClosure started")


# ── Main loop ─────────────────────────────────────────────────────────────────
print("\nMap v9  (threaded + loop closure)  —  Q to quit\n")

global_pose  = np.eye(4)
velocity     = np.eye(4)
frame_idx    = 0
_traj_win    = deque(maxlen=TRAJ_SMOOTH_WIN)
n_inliers    = 0

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

        # ── Seed ──────────────────────────────────────────────────────────────
        if prev_gray is None:
            refresh_tracks(gray, depth_raw, global_pose)
            last_kf_pose    = global_pose.copy()
            frames_since_kf = 0
            prev_gray       = gray
            print(f"  [seed]  {len(track_pts2d)} tracked pts")
            continue

        # ── LK tracking ───────────────────────────────────────────────────────
        new_pose  = None
        n_inliers = 0

        if len(track_pts2d) >= MIN_MATCHES:
            new_pts, status, err = cv2.calcOpticalFlowPyrLK(
                prev_gray, gray, track_pts2d, None, **lk_params)
            good = status.ravel()==1
            if err is not None:
                good &= err.ravel() < LK_MAX_ERR
            if good.sum() >= MIN_MATCHES:
                track_pts2d = new_pts[good]
                track_pts3d = track_pts3d[good]
                p2 = track_pts2d.reshape(-1,2).astype(np.float64)
                p3 = track_pts3d.astype(np.float64)
                new_pose, n_inliers = _pnp(p3, p2)
            else:
                track_pts2d = new_pts[good] if good.any() else np.zeros((0,1,2),np.float32)
                track_pts3d = track_pts3d[good] if good.any() else np.zeros((0,3),np.float64)

        # ── ORB fallback ──────────────────────────────────────────────────────
        if new_pose is None:
            new_pose, n_inliers = orb_fallback(gray, depth_raw)
            if new_pose is not None:
                print(f"  [fallback]  frame {frame_idx}  inliers={n_inliers}")
                track_pts2d = np.zeros((0,1,2),np.float32)
                track_pts3d = np.zeros((0,3),  np.float64)
                refresh_tracks(gray, depth_raw, new_pose)

        # ── Update pose ───────────────────────────────────────────────────────
        if new_pose is not None:
            velocity  = np.linalg.inv(global_pose) @ new_pose
            vel_trans = np.linalg.norm(velocity[:3,3])
            alpha     = float(np.clip(vel_trans/0.05, 0.1, 0.9))
            new_pose[:3,3] = alpha*new_pose[:3,3] + (1-alpha)*global_pose[:3,3]
            global_pose = new_pose
        else:
            global_pose = global_pose @ velocity
            velocity    = velocity*0.5 + np.eye(4)*0.5
            print(f"  [skip]  frame {frame_idx}")

        # ── Append trajectory ─────────────────────────────────────────────────
        with traj_lock:
            trajectory.append(global_pose[:3,3].copy())
            _traj_win.append(global_pose[:3,3].copy())
            smooth_traj.append(np.mean(_traj_win, axis=0))

        frames_since_kf += 1

        # ── Tracker refresh ───────────────────────────────────────────────────
        if len(track_pts2d) < MIN_TRACK:
            refresh_tracks(gray, depth_raw, global_pose)

        # ── Keyframe → mapping thread ─────────────────────────────────────────
        if needs_kf(global_pose):
            with traj_lock:
                ti = len(trajectory) - 1
            try:
                map_queue.put_nowait((gray.copy(), depth_raw.copy(),
                                     global_pose.copy(), ti))
                last_kf_pose    = global_pose.copy()
                frames_since_kf = 0
            except queue.Full:
                pass

        # ── Apply loop closure correction ──────────────────────────────────────
        if not corr_queue.empty():
            try:
                corr = corr_queue.get_nowait()
                drift      = corr['drift_pos']
                idx_start  = corr['traj_start_idx']
                idx_end    = corr['traj_end_idx']

                with traj_lock:
                    n_span = max(1, idx_end - idx_start)
                    for i in range(idx_start, min(len(trajectory), idx_end+1)):
                        t = (i - idx_start) / n_span
                        trajectory[i]  = trajectory[i]  + t * drift
                    # Rebuild smooth_traj from corrected trajectory
                    smooth_traj.clear(); _traj_win.clear()
                    for pos in trajectory:
                        _traj_win.append(pos)
                        smooth_traj.append(np.mean(_traj_win, axis=0))

                # Correct current pose
                global_pose[:3,3] += drift

                # Reset tracker — 3D points are now stale after correction
                track_pts2d = np.zeros((0,1,2), np.float32)
                track_pts3d = np.zeros((0,3),   np.float64)

                print(f"  [correction applied]  drift={np.linalg.norm(drift):.3f}m")
            except queue.Empty:
                pass

        # ── 2D render ─────────────────────────────────────────────────────────
        if frame_idx % RENDER_EVERY == 0:
            cv2.imshow("Map v9 — 2D top-down", render_2d())

        # ── Camera feed overlay ───────────────────────────────────────────────
        dbg = cv2.cvtColor(color_raw, cv2.COLOR_RGB2BGR)
        for pt in track_pts2d.reshape(-1,2):
            cv2.circle(dbg,(int(pt[0]),int(pt[1])),3,(0,255,0),-1)
        with map_lock:
            mn = _map_n
        with traj_lock:
            tn = len(trajectory)
        cv2.putText(dbg, f"tracked:{len(track_pts2d)}  map:{mn}  traj:{tn}  f:{frame_idx}",
                    (8,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        cv2.putText(dbg, f"inliers:{n_inliers}  "
                         f"map_q:{map_queue.qsize()}  lc_q:{lc_queue.qsize()}",
                    (8,40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,220,255), 1)
        cv2.imshow("map_v9 — camera", dbg)

        if cv2.waitKey(1) & 0xFF == ord('q'): break

        prev_gray = gray

finally:
    running.clear()
    t_map.join(timeout=3)
    t_lc.join(timeout=3)

    with map_lock:
        pts_save = _map_pts().copy()
        n_save   = _map_n
    with traj_lock:
        traj_save = list(trajectory)

    if n_save > 0:
        np.save("map_v9_pts.npy", pts_save)
    if traj_save:
        np.save("map_v9_trajectory.npy", np.array(traj_save))

    print(f"\n[saved] map_v9_pts.npy         — {n_save:,} points")
    print(f"[saved] map_v9_trajectory.npy  — {len(traj_save)} poses")
    pipeline.stop()
    cv2.destroyAllWindows()
    print("[done]")
