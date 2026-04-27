"""
view_map_v7.py
--------------
Open3D 3D viewer for map_v14 output.

Shows:
  grey points  — 3D map point cloud
  blue line    — camera trajectory
  green sphere — start position
  orange sphere— end position
  XYZ frame    — world origin (X=red  Y=green  Z=blue)

Usage:
    python view_map_v7.py                        # loads map_v14_pts + map_v14_trajectory
    python view_map_v7.py mypts.npy mytraj.npy   # custom files

Controls:
    Left drag   — rotate
    Right drag  — pan
    Scroll      — zoom
    R           — reset view
    Q / Esc     — quit
"""

import numpy as np
import open3d as o3d
import sys
import os


# ── Load files ────────────────────────────────────────────────────────────────
pts_file  = sys.argv[1] if len(sys.argv) > 1 else "map_v14_pts.npy"
traj_file = sys.argv[2] if len(sys.argv) > 2 else "map_v14_trajectory.npy"

if not os.path.exists(pts_file):
    print(f"[error]  {pts_file} not found — run map_v14.py first")
    sys.exit(1)

pts  = np.load(pts_file).astype(np.float64)
traj = np.load(traj_file).astype(np.float64) if os.path.exists(traj_file) else None

print(f"[loaded]  {len(pts):,} map points  ({pts_file})")
if traj is not None:
    path_len = float(np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1)))
    print(f"[loaded]  {len(traj)} trajectory poses  —  path length {path_len:.2f} m")
else:
    print(f"[warn]    {traj_file} not found — showing map only")


# ── Point cloud ───────────────────────────────────────────────────────────────
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pts)
pcd.colors = o3d.utility.Vector3dVector(
    np.tile([0.65, 0.65, 0.65], (len(pts), 1)))

geoms = [pcd]


# ── Trajectory + start/end markers ───────────────────────────────────────────
if traj is not None and len(traj) >= 2:
    # Blue trajectory line
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(traj)
    lines = [[i, i+1] for i in range(len(traj)-1)]
    line_set.lines  = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(
        np.tile([0.0, 0.3, 1.0], (len(lines), 1)))
    geoms.append(line_set)

    # Green sphere — start
    s_start = o3d.geometry.TriangleMesh.create_sphere(radius=0.10)
    s_start.translate(traj[0])
    s_start.paint_uniform_color([0.0, 0.9, 0.1])
    s_start.compute_vertex_normals()
    geoms.append(s_start)

    # Orange sphere — end
    s_end = o3d.geometry.TriangleMesh.create_sphere(radius=0.10)
    s_end.translate(traj[-1])
    s_end.paint_uniform_color([1.0, 0.4, 0.0])
    s_end.compute_vertex_normals()
    geoms.append(s_end)

    print(f"[info]    start {traj[0].round(2)}  →  end {traj[-1].round(2)}")
    print(f"[info]    drift start→end: "
          f"{np.linalg.norm(traj[-1]-traj[0]):.3f} m")


# ── World origin frame ────────────────────────────────────────────────────────
origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
geoms.append(origin)


# ── Display ───────────────────────────────────────────────────────────────────
print("\nControls: left-drag=rotate  right-drag=pan  scroll=zoom  R=reset  Q=quit\n")

o3d.visualization.draw_geometries(
    geoms,
    window_name="SLAM Map v7 — 3D",
    width=1280,
    height=800,
    point_show_normal=False)
