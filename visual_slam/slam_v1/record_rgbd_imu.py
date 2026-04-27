"""
record_rgbd_imu.py
------------------
Records RGB + Depth to a RealSense .bag file AND saves IMU yaw from
MPU-9250 (via ESP32 serial) to a matching CSV file.

Output files (same base name):
    recordings/YYYY-MM-DD_HH-MM-SS.bag       — RGB + depth (RealSense)
    recordings/YYYY-MM-DD_HH-MM-SS_imu.csv   — frame_idx, yaw_deg

CSV format:
    frame_idx,yaw_deg
    0,0.000
    1,0.123
    ...

During playback in map_v12.py, set:
    PLAYBACK_FILE = r"recordings/YYYY-MM-DD_HH-MM-SS.bag"
    IMU_CSV       = r"recordings/YYYY-MM-DD_HH-MM-SS_imu.csv"

Wiring:
    MPU-9250 → ESP32 GPIO 21/22 → USB serial to PC

Usage:
    python record_rgbd_imu.py              → auto-named files
    python record_rgbd_imu.py mywalk       → recordings/mywalk.bag + mywalk_imu.csv

Controls:
    Q / Space / Esc  — stop recording
"""

import cv2
import numpy as np
import pyrealsense2 as rs
import os
import sys
import time
import serial
from datetime import datetime


# ── Settings ──────────────────────────────────────────────────────────────────
IMU_PORT = 'COM5'     # ESP32 COM port — change to match your system
IMU_BAUD = 115200
YAW_SIGN = 1          # flip to -1 if yaw direction is reversed


# ── Output paths ──────────────────────────────────────────────────────────────
os.makedirs("recordings", exist_ok=True)

if len(sys.argv) > 1:
    base = os.path.join("recordings", sys.argv[1].replace(".bag", ""))
else:
    base = f"recordings/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

bag_file = base + ".bag"
imu_file = base + "_imu.csv"


# ── IMU — synchronous, no thread ──────────────────────────────────────────────
_ser     = None
_ser_buf = b''
_yaw     = 0.0
imu_ok   = False

try:
    _ser = serial.Serial(IMU_PORT, IMU_BAUD, timeout=0)   # non-blocking
    _ser.reset_input_buffer()
    imu_ok = True
    print(f"[IMU]  connected on {IMU_PORT}")
except Exception as e:
    print(f"[IMU]  not connected ({e})")
    print("       Recording will continue without IMU data.")


def read_yaw():
    """Drain serial buffer and return the latest yaw — non-blocking."""
    global _ser_buf, _yaw
    if _ser is None:
        return _yaw
    try:
        waiting = _ser.in_waiting
        if waiting:
            _ser_buf += _ser.read(waiting)
            lines = _ser_buf.split(b'\n')
            _ser_buf = lines[-1]
            for raw in reversed(lines[:-1]):
                line = raw.decode('utf-8', errors='ignore').strip()
                if not line or line.startswith('#'):
                    continue
                try:
                    _yaw = float(line) * YAW_SIGN
                    break
                except ValueError:
                    continue
    except Exception:
        pass
    return _yaw


# ── RealSense ──────────────────────────────────────────────────────────────────
pipeline  = rs.pipeline()
config    = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16,  30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
config.enable_record_to_file(bag_file)

profile   = pipeline.start(config)
colorizer = rs.colorizer()
align     = rs.align(rs.stream.color)

print(f"\n[REC]  bag  → {bag_file}")
print(f"[REC]  imu  → {imu_file}  ({'active' if imu_ok else 'NO IMU — zeros saved'})")
print("       Move camera slowly. Q / Space / Esc to stop.\n")


# ── Main loop ──────────────────────────────────────────────────────────────────
frame_count = 0
start_time  = None
imu_log     = []   # list of (frame_idx, yaw_deg)

try:
    while True:
        frames      = align.process(pipeline.wait_for_frames())
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            continue

        if start_time is None:
            start_time = time.time()

        # Snapshot IMU yaw at this exact frame (synchronous — no thread)
        yaw = read_yaw()
        imu_log.append((frame_count, yaw))
        frame_count += 1

        # ── Preview ───────────────────────────────────────────────────────────
        color_bgr     = cv2.cvtColor(
            np.asanyarray(color_frame.get_data()), cv2.COLOR_RGB2BGR)
        depth_colored = np.asanyarray(
            colorizer.colorize(depth_frame).get_data())

        display = np.hstack([color_bgr, depth_colored])

        elapsed = time.time() - start_time
        fps     = frame_count / elapsed if elapsed > 0 else 0
        size_mb = os.path.getsize(bag_file) / 1_048_576 if os.path.exists(bag_file) else 0

        # Status overlay
        cv2.circle(display, (18, 18), 8, (0, 0, 255), -1)
        cv2.putText(display,
                    f"REC  {elapsed:5.1f}s   {frame_count} frames   "
                    f"{fps:.1f} fps   {size_mb:.0f} MB",
                    (34, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(display,
                    f"IMU yaw: {yaw:.1f} deg   ({'active' if imu_ok else 'NO IMU'})",
                    (34, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 150) if imu_ok else (0, 100, 255), 1)
        cv2.putText(display, "Q / SPACE — stop",
                    (34, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
        cv2.putText(display, "RGB",   (10,  90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(display, "DEPTH", (650, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("Recording  —  RGB | Depth | IMU", display)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord(' '), 27):
            break

finally:
    if _ser is not None:
        try:
            _ser.close()
        except Exception:
            pass
    pipeline.stop()
    cv2.destroyAllWindows()

    # ── Save IMU CSV ──────────────────────────────────────────────────────────
    with open(imu_file, 'w') as f:
        f.write("frame_idx,yaw_deg\n")
        for idx, yaw in imu_log:
            f.write(f"{idx},{yaw:.4f}\n")

    elapsed = time.time() - start_time if start_time else 0
    size_mb = os.path.getsize(bag_file) / 1_048_576 if os.path.exists(bag_file) else 0

    print(f"\n[saved]  {bag_file}")
    print(f"         {frame_count} frames  |  {elapsed:.1f}s  |  {size_mb:.1f} MB")
    print(f"[saved]  {imu_file}  ({len(imu_log)} rows)")
    print(f"\nTo process:")
    print(f'    PLAYBACK_FILE = r"{bag_file}"')
    print(f'    IMU_CSV       = r"{imu_file}"')
