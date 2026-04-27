"""
view_trail.py
-------------
Load a saved trail_v15.npy and display a smoothed camera path in Open3D.

Usage:
    python view_trail.py              # loads trail_v15.npy
    python view_trail.py my_trail.npy
"""

import sys
import numpy as np
import open3d as o3d

SMOOTH_WIN = 9    # moving-average window — increase for smoother, decrease for tighter

trail_file = sys.argv[1] if len(sys.argv) > 1 else "trail_v15.npy"

trail = np.load(trail_file)
print(f"[trail]  {trail_file}  —  {len(trail)} poses")


def smooth(pts, w):
    out = np.zeros_like(pts)
    for i in range(len(pts)):
        lo = max(0, i - w // 2)
        hi = min(len(pts), i + w // 2 + 1)
        out[i] = pts[lo:hi].mean(axis=0)
    return out


smoothed = smooth(trail, SMOOTH_WIN)
path_len = float(np.sum(np.linalg.norm(np.diff(smoothed, axis=0), axis=1)))
print(f"[trail]  path length: {path_len:.2f} m")
print(f"[trail]  start {smoothed[0].round(3)}  →  end {smoothed[-1].round(3)}")

# Trail line
line_set = o3d.geometry.LineSet()
line_set.points = o3d.utility.Vector3dVector(smoothed)
line_set.lines  = o3d.utility.Vector2iVector(
    [[i, i + 1] for i in range(len(smoothed) - 1)])
line_set.colors = o3d.utility.Vector3dVector(
    np.tile([1.0, 0.0, 0.0], (len(smoothed) - 1, 1)))

# Start sphere (green)
start = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
start.translate(smoothed[0])
start.paint_uniform_color([0.0, 0.9, 0.1])
start.compute_vertex_normals()

# End sphere (orange)
end = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
end.translate(smoothed[-1])
end.paint_uniform_color([1.0, 0.4, 0.0])
end.compute_vertex_normals()

geoms = [
    line_set,
    start,
    end,
    o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3),
]

o3d.visualization.draw_geometries(
    geoms,
    window_name="Trail viewer",
    width=1280,
    height=720,
)
