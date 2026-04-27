# Visual SLAM — Memory-Based Navigation with Intel RealSense D435

A ground-up implementation of visual SLAM (Simultaneous Localisation and Mapping) built for memory-based navigation using an Intel RealSense D435 RGBD camera and an ESP32 + MPU-9250 IMU. The goal is a system that can build a persistent 3D map of an environment and later re-use that map to localise a camera and plan paths — similar to what RTAB-Map does, built from scratch.

---

## Project Goal

> Walk through a room once → build a map → walk through again → camera knows exactly where it is at every step.

The long-term target is a lightweight, dependency-minimal SLAM pipeline that runs on a Windows laptop with a commodity depth camera, produces a metric-scale 3D feature map, and supports robust re-localisation against that map for navigation.

---

## Hardware

| Component | Details |
|---|---|
| **Depth camera** | Intel RealSense D435 (640×480, 30 fps, depth + RGB) |
| **IMU** | MPU-9250 on ESP32 DevKit (Madgwick filter, 100 Hz yaw over serial) |
| **Host** | Windows 11, USB 3.0 |

The IMU is optional — all SLAM scripts work with camera-only input. IMU yaw is used as a supplementary heading estimate.

---

## What Has Been Built

The project is structured as an iterative series of versions, each adding one concept at a time. This is intentional — every version is a runnable study script, not a throwaway draft.

```
slam/
├── esp32_mpu9250_madgwick/      ← ESP32 IMU firmware (Arduino)
├── visual_slam/
│   ├── recordings/              ← RealSense .bag files + IMU .csv files
│   ├── milestones/              ← Snapshot of key working versions
│   ├── slam_v1/                 ← Iterative development (v5 → v17)
│   └── slam_v2/                 ← Current clean architecture
│       ├── map_v17_base.py      ← ORB map-building + PnP tracking (standalone)
│       ├── map_v18.py           ← Study: features → 3D world map + keyframes + FLANN
│       ├── map_builder.py       ← Phase 1: build map + bundle adjustment → map.npz
│       ├── localizer.py         ← Phase 2: localise against map.npz
│       └── motion_log.py        ← LK optical flow → human-readable motion description
└── SLAM_IMPROVEMENT_NOTES.txt   ← Design notes and roadmap
```

### Version History (slam_v1/)

| Version | What was added |
|---|---|
| v5 | ORB tracking, local map (8 keyframes), motion model, voxel dedup |
| v10 | Rotation cap, 3-thread architecture (main / mapping / loop closure) |
| v13–v15 | Loop closure experiments, bundle adjustment experiments |
| v16 | Relaxed matching thresholds, improved RANSAC |
| v17 | Final standalone ORB map-builder, FLANN keyframe database |

### Current Architecture (slam_v2/)

```
RealSense .bag
      │
      ├─ Color (RGB) ──────────────► ORB detect (1500 kp/frame)
      │                                     │
      └─ Depth (uint16) ─► depth_at()       │
                                   │        │
                                   └──► unproject(u,v,z) → p_cam
                                                │
                                       camera_pose (SE3)
                                       [from PnP RANSAC]
                                                │
                                       p_world = R·p_cam + t
                                                │
                            ┌───────────────────┴──────────────────┐
                            │  Persistent 3D map                   │
                            │  pt_id → world_xyz (never removed)   │
                            └──────────────────────────────────────┘
                                                │
                      ┌─────────────────────────┴──────────────────────┐
                      │  Keyframe DB (every 15 cm or 10°)              │
                      │  descriptor matrix → FLANN LSH index           │
                      │  trainIdx → map point ID (re-localisation)     │
                      └────────────────────────────────────────────────┘
```

**Pose estimation — merged matching:**
1. **FLANN global match** — current descriptors vs ALL keyframe descriptors → map point IDs directly (re-localisation, loop closure signal)
2. **Frame-to-frame carry-forward** — fills gaps between keyframes for fresh map points
3. **PnP RANSAC** — merged 2D↔3D correspondences → SE3 camera pose

---

## Algorithms Used

| Algorithm | Purpose | Implementation |
|---|---|---|
| ORB | Feature detection + 256-bit binary descriptors | `cv2.ORB_create(nfeatures=1500)` |
| BFMatcher (Hamming) | Frame-to-frame descriptor matching | `cv2.BFMatcher(NORM_HAMMING)` |
| Lowe ratio test | Outlier rejection | threshold 0.85 |
| PnP RANSAC | 6-DOF pose from 2D↔3D correspondences | `cv2.solvePnPRansac` (200 iter, 4px) |
| FLANN LSH | Fast binary descriptor database search | `cv2.FlannBasedMatcher` (algorithm=6) |
| ICP (point-to-plane) | Frame-to-frame odometry fallback | `open3d.pipelines.registration` |
| Bundle adjustment | Joint pose + point optimisation | `scipy.optimize.least_squares` (Huber, sparse) |
| LK optical flow | Lightweight frame-to-frame feature tracking | `cv2.calcOpticalFlowPyrLK` |
| Shi-Tomasi corners | Feature seeding for LK | `cv2.goodFeaturesToTrack` |
| Madgwick filter | IMU quaternion integration (firmware) | ESP32 Arduino library |

---

## Roadmap

```
✅  Phase 0  ORB detection + depth lifting → 3D world map
✅  Phase 1  PnP pose estimation → camera trail
✅  Step A   Keyframe selection (15 cm / 10° thresholds)
✅  Step B   FLANN descriptor database → global matching + re-localisation
⬜  Step C   Loop closure detection (FLANN match to old KF + PnP inlier count)
⬜  Step D   Pose graph optimisation (g2o or GTSAM — distribute loop drift)
⬜  Step E   Map save / load + localisation-only mode (no new points)
⬜  Step F   2D occupancy grid from 3D map (floor-level slice)
⬜  Step G   Path planning on occupancy grid (A* / RRT)
```

Expected accuracy milestones:

| State | Drift on 10 m loop |
|---|---|
| Current (no loop closure) | 1–3 m |
| After Step C+D (loop closure + graph opt) | 0.1–0.3 m |
| After Step E (BA + optimised map) | 0.05–0.15 m |

---

## Installation

### 1. Clone

```bash
git clone https://github.com/YOUR_USERNAME/visual-slam.git
cd visual-slam
```

### 2. Python environment

Python 3.9–3.11 recommended (pyrealsense2 wheel availability).

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Intel RealSense SDK

The Python wheel `pyrealsense2` is included in `requirements.txt`, but the underlying **librealsense** runtime must also be installed:

- **Windows:** Download and run the [Intel RealSense SDK 2.0 installer](https://github.com/IntelRealSense/librealsense/releases) (choose the `.exe` for your version).
- **Ubuntu:** Follow the [official APT instructions](https://github.com/IntelRealSense/librealsense/blob/master/doc/distribution_linux.md).

### 4. (Optional) ESP32 IMU firmware

Open `esp32_mpu9250_madgwick/esp32_mpu9250_madgwick.ino` in Arduino IDE.
Install board package: **ESP32 by Espressif Systems**.
Install libraries: **MPU9250** (hideakitai), **MadgwickAHRS**.
Flash to ESP32, connect via USB — default baud 115200, port COM5 (edit `record_rgbd_imu.py` if different).

---

## Usage

### Record a new session

```bash
cd visual_slam/slam_v1

# RGB + Depth only
python record_rgbd.py

# RGB + Depth + IMU yaw
python record_rgbd_imu.py
```

Recordings are saved to `visual_slam/recordings/YYYY-MM-DD_HH-MM-SS.bag` (+ `_imu.csv` for IMU).

---

### Run the live feature study (map_v18)

The main current script — builds a growing 3D world map while studying ORB feature quality.

```bash
cd visual_slam/slam_v2
python map_v18.py
```

Edit line 22 in `map_v18.py` to point to your bag file:
```python
PLAYBACK_FILE = r"..\recordings\YOUR_FILE.bag"
```

**Two windows open:**

| Window | Shows |
|---|---|
| OpenCV | Live color frame with ORB keypoints (blue→red by response strength), match lines |
| Open3D | 3D world map — green=tracked, orange=new, blue-grey=established; red trail=camera path; blue dots=keyframes |

**HUD:**
```
Frame 0042 | KP: 1287 | Map: 8341 | KF:  7 | FLANN:  412  FT:  388 | New:  87 | Q=quit
```

Press **Q** to quit.

---

### Two-phase mapping + localisation (v17 pipeline)

```bash
cd visual_slam/slam_v2

# Phase 1 — build map and run bundle adjustment
python map_builder.py
# → produces map.npz

# Phase 2 — localise against the saved map
python localizer.py
# → produces trail_localizer.npy
```

---

### Standalone ORB map builder (map_v17_base)

Faster iteration, no bundle adjustment, real-time Open3D visualisation.

```bash
cd visual_slam/slam_v2
python map_v17_base.py
# → produces map_v17.ply  +  trail_v17.npy
```

---

### Motion description utility

Human-readable motion log from any bag file.

```bash
cd visual_slam/slam_v2
python motion_log.py
# prints: "strong-forward | yaw-left | slight-up"
```

---

### Validation — compare two bag runs of the same scene

```bash
cd visual_slam/slam_v1
python compare_bags.py
```

Builds map from BAG_1, localises BAG_2 against it, overlays both trails in Open3D.

---

## Camera Intrinsics (RealSense D435 @ 640×480)

```python
FX = FY = 384.327880859375
CX = 321.8272705078125
CY = 239.01609802246094
```

Run `slam_v1/intel_real_sense_focul_point.py` with your camera attached to read live intrinsics.

---

## Key Design Decisions

**Why ORB over SIFT/SuperPoint?**
ORB is binary (fast Hamming matching), patent-free, and runs in real time on CPU. For the current study phase, matching quality (~50% frame-to-frame) is sufficient. SuperPoint would improve matching but adds a GPU dependency.

**Why depth-assisted instead of monocular?**
The RealSense D435 provides metric depth, eliminating the scale ambiguity that makes monocular SLAM hard. Every feature is immediately placed at a metrically correct world position.

**Why not use RTAB-Map directly?**
Building from scratch to understand every component — loop closure, BA, pose graphs — before adding them. Each version is a study script, not a production system.

**Why persistent map (no culling)?**
For navigation you need the full historical map. Points added from any viewpoint must remain available for re-localisation from that viewpoint later.

**Why FLANN over pure BFMatcher for the keyframe database?**
BFMatcher is O(N·M). As the keyframe database grows (N descriptors), FLANN's LSH index keeps query time sub-linear, which is essential for real-time re-localisation against a large map.

---

## Known Limitations

- **No loop closure yet** — drift accumulates on long trajectories (1–3 m on 10 m loop)
- **Pose graph not implemented** — no way to distribute correction after a loop is detected
- **Hardcoded intrinsics** — must edit constants if switching cameras
- **Windows / RealSense threading** — daemon threads break bag recording; all reads are synchronous serial (documented in `SLAM_IMPROVEMENT_NOTES.txt`)
- **No occupancy map** — 3D feature cloud only; no floor-plan slice yet

---

## File Reference

| File | Purpose |
|---|---|
| `slam_v2/map_v18.py` | Main study script — ORB + 3D map + keyframes + FLANN |
| `slam_v2/map_v17_base.py` | Standalone ORB map builder + PnP tracker |
| `slam_v2/map_builder.py` | Two-phase: map build + full bundle adjustment |
| `slam_v2/localizer.py` | Localise against pre-built `map.npz` |
| `slam_v2/motion_log.py` | LK optical flow → motion description |
| `slam_v1/record_rgbd.py` | Record RealSense bag |
| `slam_v1/record_rgbd_imu.py` | Record bag + IMU yaw CSV |
| `slam_v1/compare_bags.py` | Map BAG_1, localise BAG_2, overlay trails |
| `slam_v1/view_map.py` | Visualise saved point cloud + trail |
| `slam_v1/intel_real_sense_focul_point.py` | Print live camera intrinsics |
| `esp32_mpu9250_madgwick/*.ino` | ESP32 IMU firmware |
| `SLAM_IMPROVEMENT_NOTES.txt` | Full technical notes + improvement roadmap |

---

## Contributing

This is an active research/learning project. Issues and PRs are welcome, especially for:
- Loop closure implementation (Step C)
- Pose graph optimisation (Step D)
- GTSAM integration
- Occupancy grid generation (Step F)
