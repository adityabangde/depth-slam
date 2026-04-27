"""
view_map_v6.py
--------------
Viewer for map_v6 output — no Open3D required.

Shows a 2-D top-down (bird's-eye) plot:
  grey dots   — map points (X-Z floor projection)
  red  line   — camera trajectory (smoothed, WIN=15)
  green dot   — start position
  yellow dot  — end position

Usage:
    python view_map_v6.py                              # loads map_v6_pts.npy + map_v6_trajectory.npy
    python view_map_v6.py mypts.npy mytraj.npy         # custom files

Controls: scroll = zoom   right-click drag = pan   Q = quit
"""

import sys
import numpy as np
import cv2


# ── File paths ────────────────────────────────────────────────────────────────
pts_file  = sys.argv[1] if len(sys.argv) > 1 else "map_v6_pts.npy"
traj_file = sys.argv[2] if len(sys.argv) > 2 else "map_v6_trajectory.npy"


# ── Load ──────────────────────────────────────────────────────────────────────
map_pts = np.load(pts_file)
print(f"[map]   {pts_file}  —  {len(map_pts):,} points")

try:
    traj = np.load(traj_file)
    print(f"[traj]  {traj_file}  —  {len(traj)} poses")
except FileNotFoundError:
    print(f"[warn]  {traj_file} not found — showing map only")
    traj = None


# ── Smooth trajectory ─────────────────────────────────────────────────────────
WIN = 15
def _smooth(arr, w):
    if len(arr) <= w:
        return arr
    k   = np.ones(w) / w
    pad = np.pad(arr, ((w//2, w//2), (0,0)), mode='edge')
    return np.column_stack([np.convolve(pad[:,i], k, mode='valid')[:len(arr)]
                            for i in range(3)])

traj_smooth = _smooth(traj, WIN) if traj is not None else None

if traj is not None:
    total = float(np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1)))
    print(f"[traj]  total path: {total:.2f} m")
    print(f"[traj]  start {traj_smooth[0]}  →  end {traj_smooth[-1]}")


# ── Canvas settings ───────────────────────────────────────────────────────────
CANVAS_W  = 1000
CANVAS_H  = 1000
MAP_SCALE = 100       # pixels per metre
OX        = CANVAS_W // 2
OZ        = CANVAS_H // 2

# Pan/zoom state
zoom   = 1.0
pan_x  = 0
pan_z  = 0
drag   = False
drag_start = (0, 0)
pan_start  = (0, 0)


def world_to_px(x, z):
    col = int(OX + (x - pan_x) * MAP_SCALE * zoom)
    row = int(OZ - (z - pan_z) * MAP_SCALE * zoom)
    return col, row


def build_frame():
    canvas = np.full((CANVAS_H, CANVAS_W, 3), 15, dtype=np.uint8)

    # Grid lines every 1 m
    for m in range(-10, 11):
        gx, _ = world_to_px(m, 0);  _, gz = world_to_px(0, m)
        col = (35, 35, 35)
        cv2.line(canvas, (gx, 0), (gx, CANVAS_H), col, 1)
        cv2.line(canvas, (0, gz), (CANVAS_W, gz), col, 1)

    # Origin
    ox, oz = world_to_px(0, 0)
    cv2.line(canvas, (ox, 0), (ox, CANVAS_H), (55, 55, 55), 1)
    cv2.line(canvas, (0, oz), (CANVAS_W, oz), (55, 55, 55), 1)
    cv2.circle(canvas, (ox, oz), 5, (90, 90, 90), -1)

    # Map points
    if len(map_pts) > 0:
        px = (OX + (map_pts[:, 0] - pan_x) * MAP_SCALE * zoom).astype(np.int32)
        pz = (OZ - (map_pts[:, 2] - pan_z) * MAP_SCALE * zoom).astype(np.int32)
        valid = (px >= 0) & (px < CANVAS_W) & (pz >= 0) & (pz < CANVAS_H)
        canvas[pz[valid], px[valid]] = (140, 140, 140)

    # Trajectory
    if traj_smooth is not None and len(traj_smooth) >= 2:
        tpx = (OX + (traj_smooth[:, 0] - pan_x) * MAP_SCALE * zoom).astype(np.int32)
        tpz = (OZ - (traj_smooth[:, 2] - pan_z) * MAP_SCALE * zoom).astype(np.int32)
        pts_l = np.stack([tpx, tpz], axis=1).reshape(-1, 1, 2)
        cv2.polylines(canvas, [pts_l], False, (0, 0, 220), 2)

        # Start — green
        sx, sz = world_to_px(traj_smooth[0, 0], traj_smooth[0, 2])
        cv2.circle(canvas, (sx, sz), 8, (0, 200, 0), -1)

        # End — yellow
        ex, ez = world_to_px(traj_smooth[-1, 0], traj_smooth[-1, 2])
        cv2.circle(canvas, (ex, ez), 8, (0, 220, 220), -1)

    # Legend
    cv2.putText(canvas, f"map: {len(map_pts):,} pts   traj: {len(traj) if traj is not None else 0} poses",
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 1)
    cv2.putText(canvas, "scroll=zoom  right-drag=pan  Q=quit",
                (10, CANVAS_H-10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), 1)
    cv2.putText(canvas, f"zoom={zoom:.1f}x",
                (CANVAS_W-100, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1)

    return canvas


# ── Mouse handler ─────────────────────────────────────────────────────────────
def on_mouse(event, x, y, flags, _):
    global zoom, pan_x, pan_z, drag, drag_start, pan_start
    if event == cv2.EVENT_MOUSEWHEEL:
        factor = 1.15 if flags > 0 else 1/1.15
        zoom   = float(np.clip(zoom * factor, 0.1, 20.0))
    elif event == cv2.EVENT_RBUTTONDOWN:
        drag = True; drag_start = (x, y); pan_start = (pan_x, pan_z)
    elif event == cv2.EVENT_RBUTTONUP:
        drag = False
    elif event == cv2.EVENT_MOUSEMOVE and drag:
        dx = (x - drag_start[0]) / (MAP_SCALE * zoom)
        dz = (y - drag_start[1]) / (MAP_SCALE * zoom)
        pan_x = pan_start[0] - dx
        pan_z = pan_start[1] + dz


# ── Show ──────────────────────────────────────────────────────────────────────
cv2.namedWindow("Map v6 — 2D top-down", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Map v6 — 2D top-down", on_mouse)

print("\nControls: scroll=zoom   right-click drag=pan   Q=quit\n")

while True:
    cv2.imshow("Map v6 — 2D top-down", build_frame())
    key = cv2.waitKey(30) & 0xFF
    if key in (ord('q'), 27):
        break

cv2.destroyAllWindows()
print("[done]")
