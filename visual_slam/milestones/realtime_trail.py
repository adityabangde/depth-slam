"""
realtime_trail.py
-----------------
Live camera — pose estimation only, draws the trail in real time.
No point cloud building. No voxel checks. No carving.
Just FAST + ORB + PnP → camera position → red line in Open3D.

Q in Open3D window to quit.
"""

import numpy as np
import cv2
import pyrealsense2 as rs
import open3d as o3d
from collections import deque


# ── Settings ──────────────────────────────────────────────────────────────────
FRAME_W       = 640
FRAME_H       = 480
FPS           = 30
MIN_DEPTH_M   = 0.3
MAX_DEPTH_M   = 4.0
ORB_FEATURES  = 500       # lower than map_v5 — just enough for tracking
MAX_KF        = 8
KF_TRANS      = 0.12
KF_ROT_DEG    = 12.0
MIN_KF_FRAMES = 5
SEARCH_RADIUS = 30
MAX_HAMMING   = 60
MIN_MATCHES   = 12
POS_ALPHA     = 0.5       # position EMA: lower = smoother, slightly more lag on real movement


# ── Hamming LUT ───────────────────────────────────────────────────────────────
_POPCOUNT = np.array([bin(i).count('1') for i in range(256)], dtype=np.int32)
def hamming(a, b): return int(_POPCOUNT[np.bitwise_xor(a, b)].sum())


# ── RealSense ─────────────────────────────────────────────────────────────────
pipeline = rs.pipeline()
cfg      = rs.config()
cfg.enable_stream(rs.stream.depth, FRAME_W, FRAME_H, rs.format.z16,  FPS)
cfg.enable_stream(rs.stream.color, FRAME_W, FRAME_H, rs.format.rgb8, FPS)
profile  = pipeline.start(cfg)

align = rs.align(rs.stream.color)

depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
intr = (profile.get_stream(rs.stream.color)
               .as_video_stream_profile()
               .get_intrinsics())
fx, fy = intr.fx, intr.fy
cx, cy = intr.ppx, intr.ppy
K    = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], dtype=np.float64)
dist = np.zeros((4,1), dtype=np.float64)

spatial  = rs.spatial_filter()
temporal = rs.temporal_filter()


# ── Detectors ─────────────────────────────────────────────────────────────────
fast    = cv2.FastFeatureDetector_create(threshold=10, nonmaxSuppression=True)
orb     = cv2.ORB_create(nfeatures=ORB_FEATURES)
matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

def detect(gray):
    kp = fast.detect(gray, None)
    if not kp:
        return [], None
    kp = sorted(kp, key=lambda k: k.response, reverse=True)[:ORB_FEATURES]
    return orb.compute(gray, kp)


# ── Keyframe ──────────────────────────────────────────────────────────────────
class KF:
    __slots__ = ('pose','kp','des','pts3d')
    def __init__(self, pose, kp, des, pts3d):
        self.pose=pose; self.kp=kp; self.des=des; self.pts3d=pts3d

local_map = deque(maxlen=MAX_KF)
_kf_id    = 0

def make_kf(pose, gray, depth_raw, kp=None, des=None):
    global _kf_id
    if kp is None: kp, des = detect(gray)
    if des is None or not kp: return None
    H, W = depth_raw.shape
    WIN  = 3
    pts3d = np.full((len(kp),3), np.nan)
    for i, k in enumerate(kp):
        u0,v0 = int(k.pt[0]), int(k.pt[1])
        patch = depth_raw[max(0,v0-WIN):min(H,v0+WIN+1),
                          max(0,u0-WIN):min(W,u0+WIN+1)].astype(np.float32)*depth_scale
        valid = patch[(patch>MIN_DEPTH_M)&(patch<MAX_DEPTH_M)]
        if len(valid)==0: continue
        d = float(np.median(valid))
        pts3d[i] = pose[:3,:3] @ np.array([(k.pt[0]-cx)/fx*d,(k.pt[1]-cy)/fy*d,d]) + pose[:3,3]
    _kf_id += 1
    return KF(pose, kp, des, pts3d)


# ── Spatial grid ──────────────────────────────────────────────────────────────
class Grid:
    def __init__(self, kp, des, cell=30):
        self.cell=cell; self.g={}
        for i,k in enumerate(kp):
            gx,gy=int(k.pt[0]/cell),int(k.pt[1]/cell)
            self.g.setdefault((gx,gy),[]).append((i,k.pt,des[i]))
    def query(self,u,v,r):
        gx,gy=int(u/self.cell),int(v/self.cell); nc=int(r/self.cell)+1; r2=r*r
        return [item for dx in range(-nc,nc+1) for dy in range(-nc,nc+1)
                for item in self.g.get((gx+dx,gy+dy),[])
                if (item[1][0]-u)**2+(item[1][1]-v)**2<=r2]


# ── PnP ───────────────────────────────────────────────────────────────────────
def pnp(p3, p2):
    ok,rvec,tvec,inl = cv2.solvePnPRansac(p3,p2,K,dist,
                        iterationsCount=200,reprojectionError=4.0,confidence=0.99)
    if not ok or inl is None or len(inl)<MIN_MATCHES: return None,0
    R,_ = cv2.Rodrigues(rvec); T=np.eye(4)
    T[:3,:3]=R.T; T[:3,3]=-(R.T@tvec.ravel())
    return T, len(inl)

def project_local(pred_pose):
    if not local_map:
        return np.zeros((0,3)),np.zeros((0,32),dtype=np.uint8),np.zeros((0,2))
    w2c=np.linalg.inv(pred_pose); R,t=w2c[:3,:3],w2c[:3,3]
    ap,ad,au=[],[],[]
    for kf in local_map:
        v=~np.any(np.isnan(kf.pts3d),axis=1)
        if not v.any(): continue
        pts=kf.pts3d[v]; des=kf.des[v]
        pc=(R@pts.T).T+t; z=pc[:,2]; f=z>MIN_DEPTH_M
        if not f.any(): continue
        u=pc[f,0]/z[f]*fx+cx; v2=pc[f,1]/z[f]*fy+cy
        ib=(u>=0)&(u<FRAME_W)&(v2>=0)&(v2<FRAME_H)
        ap.append(pts[f][ib]); ad.append(des[f][ib])
        au.append(np.stack([u[ib],v2[ib]],axis=1))
    if not ap:
        return np.zeros((0,3)),np.zeros((0,32),dtype=np.uint8),np.zeros((0,2))
    return np.vstack(ap),np.vstack(ad),np.vstack(au)

def guided(kp,des,mp3,md,muv):
    if des is None or not len(mp3): return np.zeros((0,3)),np.zeros((0,2))
    grid=Grid(kp,des); p3,p2=[],[];  used=set()
    for pt3,dm,(up,vp) in zip(mp3,md,muv):
        cands=grid.query(up,vp,SEARCH_RADIUS)
        bd=MAX_HAMMING; bi=-1; bpt=None
        for idx,pt2d,dc in cands:
            if idx in used: continue
            d=hamming(dm,dc)
            if d<bd: bd=d; bi=idx; bpt=pt2d
        if bi>=0: used.add(bi); p3.append(pt3); p2.append(bpt)
    if not p3: return np.zeros((0,3)),np.zeros((0,2))
    return np.array(p3,np.float64),np.array(p2,np.float64)

def fallback(kp,des):
    if not local_map or des is None: return None,0
    lkf=local_map[-1]
    raw=matcher.knnMatch(des,lkf.des,k=2)
    good=[m for pair in raw if len(pair)==2
          for m,n in [(pair[0],pair[1])] if m.distance<0.75*n.distance]
    if len(good)<MIN_MATCHES: return None,0
    p3,p2=[],[]
    for m in good:
        pt=lkf.pts3d[m.trainIdx]
        if np.any(np.isnan(pt)): continue
        p3.append(pt); p2.append(kp[m.queryIdx].pt)
    if len(p3)<MIN_MATCHES: return None,0
    return pnp(np.array(p3,np.float64),np.array(p2,np.float64))

def _rot_angle(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R)-1)/2,-1,1))))

def need_kf(pose,last,n_matches,since):
    if since<MIN_KF_FRAMES: return False
    if np.linalg.norm(pose[:3,3]-last[:3,3])>KF_TRANS: return True
    if _rot_angle(pose[:3,:3].T@last[:3,:3])>KF_ROT_DEG: return True
    if n_matches<MIN_MATCHES*2: return True
    return False


# ── Open3D viewer ─────────────────────────────────────────────────────────────
vis = o3d.visualization.Visualizer()
vis.create_window("Live Trail", width=1280, height=720)

# Trail line
trail_line = o3d.geometry.LineSet()
vis.add_geometry(trail_line)

# Current camera marker — bright green dot
cam_dot = o3d.geometry.PointCloud()
cam_dot.points = o3d.utility.Vector3dVector([[0,0,0]])
cam_dot.colors = o3d.utility.Vector3dVector([[0,1,0]])
vis.add_geometry(cam_dot)

vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3))

ropt = vis.get_render_option()
ropt.point_size       = 12.0
ropt.background_color = np.array([0.05,0.05,0.05])
ropt.line_width       = 2.0


# ── Main loop ─────────────────────────────────────────────────────────────────
print("\nLive trail — move camera.  Q to quit.\n")

global_pose     = np.eye(4)
velocity        = np.eye(4)
last_kf_pose    = np.eye(4)
frames_since_kf = 0
frame_idx       = 0
view_reset      = False
trail_pts       = []

try:
    while True:

        frames      = align.process(pipeline.wait_for_frames())
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame: continue

        depth_frame = spatial.process(depth_frame)
        depth_frame = temporal.process(depth_frame)

        color_raw = np.asanyarray(color_frame.get_data())
        depth_raw = np.asanyarray(depth_frame.get_data())
        gray      = cv2.cvtColor(color_raw, cv2.COLOR_RGB2GRAY)
        kp, des   = detect(gray)

        # ── Seed ──────────────────────────────────────────────────────────────
        if not local_map:
            kf = make_kf(global_pose, gray, depth_raw, kp, des)
            if kf:
                local_map.append(kf)
                last_kf_pose    = global_pose.copy()
                frames_since_kf = 0
                trail_pts.append(global_pose[:3,3].copy())
                print("  [KF 0] seeded")
            frame_idx += 1
            continue

        # ── Pose ──────────────────────────────────────────────────────────────
        pred = global_pose @ velocity
        mp3, md, muv = project_local(pred)

        new_pose  = None
        n_inliers = 0

        if len(mp3) >= MIN_MATCHES:
            p3, p2 = guided(kp, des, mp3, md, muv)
            if len(p3) >= MIN_MATCHES:
                new_pose, n_inliers = pnp(p3, p2)

        if new_pose is None:
            new_pose, n_inliers = fallback(kp, des)

        if new_pose is not None:
            velocity = np.linalg.inv(global_pose) @ new_pose
            new_pose[:3, 3] = POS_ALPHA * new_pose[:3, 3] + (1 - POS_ALPHA) * global_pose[:3, 3]
            global_pose = new_pose
        else:
            global_pose = pred
            velocity    = velocity * 0.5 + np.eye(4) * 0.5

        trail_pts.append(global_pose[:3,3].copy())
        frames_since_kf += 1

        # ── Keyframe ──────────────────────────────────────────────────────────
        if need_kf(global_pose, last_kf_pose, n_inliers, frames_since_kf):
            kf = make_kf(global_pose, gray, depth_raw, kp, des)
            if kf:
                local_map.append(kf)
                last_kf_pose    = global_pose.copy()
                frames_since_kf = 0

        # ── Update trail ──────────────────────────────────────────────────────
        if len(trail_pts) >= 2:
            pts    = np.array(trail_pts)
            lines  = [[i, i+1] for i in range(len(pts)-1)]
            trail_line.points = o3d.utility.Vector3dVector(pts)
            trail_line.lines  = o3d.utility.Vector2iVector(lines)
            trail_line.colors = o3d.utility.Vector3dVector([[1,0,0]]*len(lines))
            vis.update_geometry(trail_line)

        # ── Move camera dot ───────────────────────────────────────────────────
        cam_dot.points = o3d.utility.Vector3dVector([global_pose[:3,3]])
        vis.update_geometry(cam_dot)

        # ── Render ────────────────────────────────────────────────────────────
        if not view_reset:
            vis.reset_view_point(True)
            view_reset = True

        if not vis.poll_events(): break
        vis.update_renderer()

        frame_idx += 1

finally:
    if trail_pts:
        np.save("trail.npy", np.array(trail_pts))
        print(f"\n[saved] trail.npy — {len(trail_pts)} positions")
    pipeline.stop()
    vis.destroy_window()
    print("[done]")
