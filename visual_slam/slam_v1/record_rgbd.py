"""
record_rgbd.py
--------------
Records RGB + Depth from D435 to a RealSense .bag file.

The .bag format stores everything:
  - Both streams (depth z16 + color rgb8)
  - All camera intrinsics
  - Timestamps
  - Device metadata

So when you play it back, it behaves exactly like a live camera —
same API calls, same intrinsics, same depth_scale. No code changes needed.

Usage:
    python record_rgbd.py              → saves to recordings/YYYY-MM-DD_HH-MM-SS.bag
    python record_rgbd.py mywalk.bag   → saves to recordings/mywalk.bag

Controls:
    Q / Space  — stop recording
"""

import cv2
import numpy as np
import pyrealsense2 as rs
import os
import sys
import time
from datetime import datetime


# ── Output path ───────────────────────────────────────────────────────────────
os.makedirs("recordings", exist_ok=True)

if len(sys.argv) > 1:
    filename = os.path.join("recordings", sys.argv[1])
    if not filename.endswith(".bag"):
        filename += ".bag"
else:
    ts       = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"recordings/{ts}.bag"


# ── RealSense — enable record to file ────────────────────────────────────────
pipeline  = rs.pipeline()
config    = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16,  30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
config.enable_record_to_file(filename)      # ← this one line enables recording

profile   = pipeline.start(config)
colorizer = rs.colorizer()                  # for depth preview only

align     = rs.align(rs.stream.color)

print(f"\n[REC]  Saving to: {filename}")
print("       Move camera slowly through the scene.")
print("       Q or SPACE to stop.\n")


# ── Main loop ─────────────────────────────────────────────────────────────────
frame_count = 0
start_time  = None

try:
    while True:
        frames      = align.process(pipeline.wait_for_frames())
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            continue

        if start_time is None:
            start_time = time.time()

        frame_count += 1

        # ── Preview ───────────────────────────────────────────────────────────
        color_bgr     = cv2.cvtColor(
            np.asanyarray(color_frame.get_data()), cv2.COLOR_RGB2BGR
        )
        depth_colored = np.asanyarray(
            colorizer.colorize(depth_frame).get_data()
        )

        display = np.hstack([color_bgr, depth_colored])

        elapsed  = time.time() - start_time
        fps      = frame_count / elapsed if elapsed > 0 else 0
        size_mb  = (os.path.getsize(filename) / 1_048_576
                    if os.path.exists(filename) else 0)

        # Red recording dot + stats
        cv2.circle(display, (18, 18), 8, (0, 0, 255), -1)
        cv2.putText(display,
                    f"REC  {elapsed:5.1f}s   {frame_count} frames   "
                    f"{fps:.1f} fps   {size_mb:.0f} MB",
                    (34, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1)
        cv2.putText(display, "Q / SPACE — stop",
                    (34, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (180, 180, 180), 1)
        cv2.putText(display, "RGB", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(display, "DEPTH", (650, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("Recording  —  RGB | Depth", display)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord(' '), 27):   # Q, Space, Escape
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()

    elapsed = time.time() - start_time if start_time else 0
    size_mb = os.path.getsize(filename) / 1_048_576 if os.path.exists(filename) else 0

    print(f"\n[saved]  {filename}")
    print(f"         {frame_count} frames  |  {elapsed:.1f}s  |  {size_mb:.1f} MB")
    print(f"\nTo process:  set PLAYBACK_FILE = r\"{filename}\"  in map_v5.py")
