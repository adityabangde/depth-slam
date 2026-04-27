"""
view_compare.py
---------------
View trail_compare_1.npy, trail_compare_2.npy, and map_compare.ply together.

RED  = bag 1 (map builder)
BLUE = bag 2 (localiser)
GREY = map point cloud

Usage:
    python view_compare.py
"""

import numpy as np
import open3d as o3d
import os

SMOOTH_WIN = 11


def smooth(pts, w):
    pts = np.array(pts, dtype=np.float64)
    out = np.zeros_like(pts)
    for i in range(len(pts)):
        lo = max(0, i - w // 2)
        hi = min(len(pts), i + w // 2 + 1)
        out[i] = pts[lo:hi].mean(axis=0)
    return out


def make_trail(pts, color):
    if len(pts) < 2:
        return None
    pts = smooth(pts, SMOOTH_WIN)
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(pts)
    ls.lines  = o3d.utility.Vector2iVector([[i, i+1] for i in range(len(pts)-1)])
    ls.colors = o3d.utility.Vector3dVector(np.tile(color, (len(pts)-1, 1)))
    return ls


def make_sphere(pos, color, radius=0.08):
    sp = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    sp.translate(pos)
    sp.paint_uniform_color(color)
    sp.compute_vertex_normals()
    return sp


geoms = [o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)]

# ── Map ───────────────────────────────────────────────────────────────────────
if os.path.exists("map_compare.ply"):
    pcd = o3d.io.read_point_cloud("map_compare.ply")
    print(f"[map]   {len(pcd.points):,} points")
    geoms.append(pcd)
else:
    print("[map]   map_compare.ply not found — run compare_bags.py first")

# ── Trail 1 (red) ─────────────────────────────────────────────────────────────
if os.path.exists("trail_compare_1.npy"):
    t1 = np.load("trail_compare_1.npy")
    print(f"[bag1]  {len(t1)} poses   start={t1[0].round(2)}  end={t1[-1].round(2)}")
    ls1 = make_trail(t1, [1.0, 0.0, 0.0])
    if ls1:
        geoms.append(ls1)
    geoms.append(make_sphere(t1[0],  [1.0, 0.5, 0.0]))   # orange = start
    geoms.append(make_sphere(t1[-1], [1.0, 0.0, 0.0]))   # red = end
else:
    print("[bag1]  trail_compare_1.npy not found")

# ── Trail 2 (blue) ────────────────────────────────────────────────────────────
if os.path.exists("trail_compare_2.npy"):
    t2 = np.load("trail_compare_2.npy")
    print(f"[bag2]  {len(t2)} poses   start={t2[0].round(2)}  end={t2[-1].round(2)}")
    ls2 = make_trail(t2, [0.0, 0.4, 1.0])
    if ls2:
        geoms.append(ls2)
    geoms.append(make_sphere(t2[0],  [0.0, 0.9, 0.1]))   # green = start
    geoms.append(make_sphere(t2[-1], [0.0, 0.4, 1.0]))   # blue = end
else:
    print("[bag2]  trail_compare_2.npy not found")

o3d.visualization.draw_geometries(
    geoms,
    window_name="Trail Comparison — RED=bag1  BLUE=bag2  GREY=map",
    width=1280,
    height=720,
)
