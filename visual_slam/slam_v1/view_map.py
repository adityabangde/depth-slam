"""
view_map.py
-----------
Visualise a saved .ply map (and optional trajectory).

Shows:
  • Point cloud   — the 3D map (colours from file, or depth-coded red→blue)
  • Blue line     — camera trajectory  (if .npy file provided)
  • Green sphere  — start position
  • Orange sphere — end position
  • Coordinate frame at origin

Usage:
    python view_map.py                          # loads map_v13.ply (no trajectory)
    python view_map.py mymap.ply                # custom map, no trajectory
    python view_map.py mymap.ply traj.npy       # map + trajectory

Controls:
    Left drag  — rotate   |   Right drag — pan   |   Scroll — zoom
    R — reset view        |   Q / Esc    — quit
"""

import sys
import os
import numpy as np
import open3d as o3d


# ── File paths ────────────────────────────────────────────────────────────────
map_file  = sys.argv[1] if len(sys.argv) > 1 else "map_v13.ply"
traj_file = sys.argv[2] if len(sys.argv) > 2 else None


# ── Load map ──────────────────────────────────────────────────────────────────
if not os.path.exists(map_file):
    print(f"[error]  {map_file} not found")
    sys.exit(1)

pcd = o3d.io.read_point_cloud(map_file)
print(f"[map]   {map_file}  —  {len(pcd.points):,} points")

# If the file has no colours, apply depth-based colouring (Z axis)
if not pcd.has_colors() and len(pcd.points) > 0:
    pts = np.asarray(pcd.points)
    z   = pts[:, 2]
    t   = np.clip((z - z.min()) / max(z.max() - z.min(), 1e-6), 0, 1)
    pcd.colors = o3d.utility.Vector3dVector(
        np.column_stack([1 - t, np.zeros(len(t)), t]))


# ── Load trajectory (optional) ────────────────────────────────────────────────
traj = None
if traj_file and os.path.exists(traj_file):
    traj = np.load(traj_file)
    print(f"[traj]  {traj_file}  —  {len(traj)} poses")
elif traj_file:
    print(f"[warn]  {traj_file} not found — showing map only")


# ── Build scene ───────────────────────────────────────────────────────────────
geoms = [pcd, o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)]

if traj is not None and len(traj) >= 2:

    # Blue trajectory line
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(traj)
    line_set.lines  = o3d.utility.Vector2iVector(
        [[i, i + 1] for i in range(len(traj) - 1)])
    line_set.colors = o3d.utility.Vector3dVector(
        np.tile([0.0, 0.3, 1.0], (len(traj) - 1, 1)))
    geoms.append(line_set)

    # Green sphere — start
    s = o3d.geometry.TriangleMesh.create_sphere(radius=0.08)
    s.translate(traj[0]); s.paint_uniform_color([0.0, 0.9, 0.1])
    s.compute_vertex_normals(); geoms.append(s)

    # Orange sphere — end
    e = o3d.geometry.TriangleMesh.create_sphere(radius=0.08)
    e.translate(traj[-1]); e.paint_uniform_color([1.0, 0.4, 0.0])
    e.compute_vertex_normals(); geoms.append(e)

    path_len = float(np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1)))
    print(f"[traj]  path length: {path_len:.2f} m")
    print(f"[traj]  start {traj[0].round(2)}  →  end {traj[-1].round(2)}")


# ── Show ──────────────────────────────────────────────────────────────────────
print("\nControls: left-drag=rotate  right-drag=pan  scroll=zoom  R=reset  Q=quit\n")

o3d.visualization.draw_geometries(
    geoms,
    window_name="SLAM Map viewer",
    width=1280,
    height=720,
)
