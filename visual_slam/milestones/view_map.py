"""
view_map.py
-----------
Visualise the saved map + camera trajectory together.

Shows:
  • Point cloud   — the 3D map (depth-coloured red→blue)
  • WHITE line    — camera path (one position per frame)
  • GREEN spheres — keyframe positions (every pose where KF was created)
  • Coordinate frame at origin

Usage:
    python view_map.py                        # loads map_v5.ply + map_v5_trajectory.npy
    python view_map.py mymap.ply mytraj.npy   # custom files
"""

import sys
import numpy as np
import open3d as o3d


# ── File paths ────────────────────────────────────────────────────────────────
map_file  = sys.argv[1] if len(sys.argv) > 1 else "map_v5.ply"
traj_file = sys.argv[2] if len(sys.argv) > 2 else "map_v5_trajectory.npy"


# ── Load map ──────────────────────────────────────────────────────────────────
pcd = o3d.io.read_point_cloud(map_file)
print(f"[map]  {map_file}  —  {len(pcd.points):,} points")


# ── Load trajectory ───────────────────────────────────────────────────────────
try:
    traj = np.load(traj_file)           # (N, 3) camera positions
    print(f"[traj] {traj_file}  —  {len(traj)} poses")
except FileNotFoundError:
    print(f"[warn] {traj_file} not found — showing map only")
    traj = None


# ── Build trajectory line (white) ─────────────────────────────────────────────
geoms = [pcd, o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)]

if traj is not None and len(traj) >= 2:

    # ── Smooth the trajectory (moving average) ────────────────────────────
    # Removes per-frame PnP jitter for cleaner visualization.
    # Does NOT change the actual map — purely cosmetic.
    WIN = 15   # smoothing window in frames; increase for smoother line
    def _smooth(arr, w):
        k = np.ones(w) / w
        pad = np.pad(arr, ((w//2, w//2), (0,0)), mode='edge')
        return np.column_stack([np.convolve(pad[:,i], k, mode='valid')[:len(arr)]
                                 for i in range(3)])

    traj_smooth = _smooth(traj, WIN) if len(traj) > WIN else traj
    traj_sub    = traj_smooth[::3]   # subsample for line density

    lines  = [[i, i+1] for i in range(len(traj_sub)-1)]
    colors = [[1.0, 0.0, 0.0]] * len(lines)   # red

    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(traj_sub)
    ls.lines  = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector(colors)
    geoms.append(ls)

    # Start marker — green sphere
    start = o3d.geometry.TriangleMesh.create_sphere(radius=0.04)
    start.translate(traj_smooth[0])
    start.paint_uniform_color([0.0, 1.0, 0.0])
    geoms.append(start)

    # End marker — yellow sphere
    end = o3d.geometry.TriangleMesh.create_sphere(radius=0.04)
    end.translate(traj_smooth[-1])
    end.paint_uniform_color([1.0, 1.0, 0.0])
    geoms.append(end)


    total_dist = float(np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1)))
    print(f"[traj] total path length: {total_dist:.2f} m")
    print(f"[traj] start {traj[0]}  →  end {traj[-1]}")


# ── Show ──────────────────────────────────────────────────────────────────────
print("\nControls: mouse=rotate  scroll=zoom  shift+drag=pan  Q=quit")

o3d.visualization.draw_geometries(
    geoms,
    window_name="Map + Trajectory",
    width=1280,
    height=720,
    zoom=0.5,
)
