"""
motion_log.py
-------------
Tracks features frame-to-frame with LK optical flow, estimates relative
camera pose each frame using depth-assisted PnP, and logs motion in plain
English: forward / backward / left / right / up / down / yaw-left /
yaw-right / pitch-up / pitch-down / roll-left / roll-right.

Camera frame convention (RealSense):
  Z = forward into scene
  X = right
  Y = down

Run:
    python motion_log.py
"""

import numpy as np
import cv2
import pyrealsense2 as rs


# ── Config ────────────────────────────────────────────────────────────────────
BAG_FILE   = r"..\recordings\2026-04-12_22-52-46.bag"
FRAME_SKIP = 1             # every frame — flow needs temporal continuity
FRAME_W, FRAME_H = 640, 480
MIN_DEPTH_M, MAX_DEPTH_M = 0.3, 8.0
DEPTH_WIN = 3              # median depth patch half-size

MAX_CORNERS    = 400       # Shi-Tomasi feature count
QUALITY_LEVEL  = 0.01
MIN_DISTANCE   = 12        # px between features
REDETECT_BELOW = 60        # re-detect when tracked points drop below this

LK_PARAMS = dict(
    winSize=(21, 21), maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

# Motion thresholds (per-frame magnitudes)
T_SLIGHT = 0.003   # m   — below: silent
T_STRONG = 0.015   # m   — above: prefix "strong"
R_SLIGHT = 0.35    # deg — below: silent
R_STRONG = 2.0     # deg — above: prefix "strong"

LOG_EVERY = 3      # print log line every N processed frames

FX = 384.327880859375
FY = 384.327880859375
CX = 321.8272705078125
CY = 239.01609802246094
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
print(f"[bag]  {BAG_FILE}\n")


# ── Helpers ───────────────────────────────────────────────────────────────────
def depth_median(depth_raw, u, v):
    u0, v0 = int(u), int(v)
    patch = depth_raw[max(0, v0 - DEPTH_WIN):min(FRAME_H, v0 + DEPTH_WIN + 1),
                      max(0, u0 - DEPTH_WIN):min(FRAME_W, u0 + DEPTH_WIN + 1)
                      ].astype(float) * depth_scale
    valid = patch[(patch > MIN_DEPTH_M) & (patch < MAX_DEPTH_M)]
    return float(np.median(valid)) if len(valid) else 0.0


def unproject(u, v, z):
    return np.array([(u - CX) * z / FX, (v - CY) * z / FY, z], dtype=np.float64)


def token(val, slight, strong, pos, neg):
    if abs(val) < slight:
        return ""
    prefix = "strong-" if abs(val) > strong else ""
    return prefix + (pos if val > 0 else neg)


def describe_motion(t_cam, rvec_rad):
    """
    t_cam    : camera translation in previous camera frame  (X=right Y=down Z=fwd)
    rvec_rad : rotation vector world(prev-cam) → curr-cam
      rvec[0] around X → pitch   (pos = nose down)
      rvec[1] around Y → yaw     (pos = turn left, Y points down so RH-rule flips)
      rvec[2] around Z → roll    (pos = roll right)
    """
    r = np.degrees(rvec_rad)
    parts = [
        token( t_cam[2], T_SLIGHT, T_STRONG, "forward",    "backward"),
        token( t_cam[0], T_SLIGHT, T_STRONG, "right",      "left"),
        token(-t_cam[1], T_SLIGHT, T_STRONG, "up",         "down"),
        token(-r[1],     R_SLIGHT, R_STRONG, "yaw-right",  "yaw-left"),
        token(-r[0],     R_SLIGHT, R_STRONG, "pitch-up",   "pitch-down"),
        token( r[2],     R_SLIGHT, R_STRONG, "roll-right",  "roll-left"),
    ]
    parts = [p for p in parts if p]
    return "  |  ".join(parts) if parts else "still"


# ── State ─────────────────────────────────────────────────────────────────────
prev_gray  = None
prev_depth = None
prev_pts   = None
frame_idx  = 0
proc_idx   = 0
last_desc  = "---"
last_inl   = 0
total_dist = 0.0

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

        color_raw = np.asanyarray(cf.get_data())
        depth_raw = np.asanyarray(df.get_data())
        gray      = cv2.cvtColor(color_raw, cv2.COLOR_RGB2GRAY)
        dbg       = cv2.cvtColor(color_raw, cv2.COLOR_RGB2BGR)

        proc_idx += 1

        # ── (Re)detect features ───────────────────────────────────────────────
        need_detect = (prev_pts is None or
                       prev_pts.shape[0] < REDETECT_BELOW)
        if need_detect:
            pts = cv2.goodFeaturesToTrack(gray, MAX_CORNERS, QUALITY_LEVEL,
                                          MIN_DISTANCE,
                                          blockSize=7)
            if pts is not None:
                prev_pts   = pts
                prev_gray  = gray.copy()
                prev_depth = depth_raw.copy()
            continue

        # ── LK optical flow ───────────────────────────────────────────────────
        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, gray, prev_pts, None, **LK_PARAMS)

        mask      = status.ravel() == 1
        good_prev = prev_pts[mask]
        good_curr = curr_pts[mask]

        # ── Lift prev 2-D → 3-D using depth at prev frame ─────────────────────
        pts3d, pts2d = [], []
        for (pu, pv), (cu, cv_) in zip(good_prev.reshape(-1, 2),
                                        good_curr.reshape(-1, 2)):
            z = depth_median(prev_depth, pu, pv)
            if z > 0:
                pts3d.append(unproject(pu, pv, z))
                pts2d.append([cu, cv_])

        # ── Relative pose via PnP ─────────────────────────────────────────────
        desc  = "still"
        n_inl = 0
        if len(pts3d) >= 8:
            ok, rvec, tvec, inliers = cv2.solvePnPRansac(
                np.array(pts3d, dtype=np.float64),
                np.array(pts2d, dtype=np.float64),
                K_MAT, DIST,
                iterationsCount=100, reprojectionError=3.0, confidence=0.99)
            if ok and inliers is not None and len(inliers) >= 6:
                n_inl  = len(inliers)
                R, _   = cv2.Rodrigues(rvec)
                t_cam  = -(R.T @ tvec.ravel())   # camera centre in prev-cam frame
                desc   = describe_motion(t_cam, rvec.ravel())
                total_dist += float(np.linalg.norm(t_cam))
                last_desc = desc
                last_inl  = n_inl

        if proc_idx % LOG_EVERY == 0:
            print(f"  f:{frame_idx:5d}  tracked:{len(good_curr):3d}"
                  f"  inliers:{n_inl:3d}  dist:{total_dist:.3f}m"
                  f"  →  {desc}")

        # ── Draw flow arrows ──────────────────────────────────────────────────
        for (pu, pv), (cu, cv_) in zip(good_prev.reshape(-1, 2),
                                        good_curr.reshape(-1, 2)):
            cv2.arrowedLine(dbg,
                            (int(pu), int(pv)), (int(cu), int(cv_)),
                            (0, 200, 180), 1, tipLength=0.4)

        # ── Overlay ───────────────────────────────────────────────────────────
        cv2.putText(dbg, last_desc,
                    (8, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 255, 80), 2)
        cv2.putText(dbg,
                    f"tracked:{len(good_curr)}  inliers:{last_inl}"
                    f"  total:{total_dist:.2f}m  f:{frame_idx}",
                    (8, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 200, 200), 1)
        cv2.imshow("motion_log", dbg)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[stopped early]"); break

        # ── Slide window forward ──────────────────────────────────────────────
        prev_gray  = gray.copy()
        prev_depth = depth_raw.copy()
        prev_pts   = good_curr.reshape(-1, 1, 2)

finally:
    pipeline.stop()
    cv2.destroyAllWindows()

print(f"\n[summary]  total distance: {total_dist:.3f} m")
print("[done]")
