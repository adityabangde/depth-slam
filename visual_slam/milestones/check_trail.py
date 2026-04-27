"""
check_trail.py
--------------
Quick viewer for a saved trail.npy file.

Usage:
    python check_trail.py              # loads trail.npy
    python check_trail.py mytrail.npy  # custom file
"""

import sys
import numpy as np
import open3d as o3d

file = sys.argv[1] if len(sys.argv) > 1 else "trail.npy"

traj = np.load(file)
print(f"[trail]  {file}  —  {len(traj)} positions")
print(f"         start : {traj[0]}")
print(f"         end   : {traj[-1]}")
total = float(np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1)))
print(f"         total distance: {total:.3f} m")

WIN = 15
def _smooth(arr, w):
    if len(arr) <= w:
        return arr
    k   = np.ones(w) / w
    pad = np.pad(arr, ((w//2, w//2), (0,0)), mode='edge')
    return np.column_stack([np.convolve(pad[:,i], k, mode='valid')[:len(arr)]
                            for i in range(3)])

traj_smooth = _smooth(traj, WIN)

geoms = [o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)]

if len(traj) >= 2:
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(traj_smooth)
    ls.lines  = o3d.utility.Vector2iVector([[i, i+1] for i in range(len(traj_smooth)-1)])
    ls.colors = o3d.utility.Vector3dVector([[1, 0, 0]] * (len(traj_smooth)-1))
    geoms.append(ls)

    # Start — green sphere
    s = o3d.geometry.TriangleMesh.create_sphere(radius=0.04)
    s.translate(traj_smooth[0]); s.paint_uniform_color([0, 1, 0])
    geoms.append(s)

    # End — yellow sphere
    e = o3d.geometry.TriangleMesh.create_sphere(radius=0.04)
    e.translate(traj_smooth[-1]); e.paint_uniform_color([1, 1, 0])
    geoms.append(e)

o3d.visualization.draw_geometries(
    geoms,
    window_name=f"Trail — {file}",
    width=1280, height=720,
)
