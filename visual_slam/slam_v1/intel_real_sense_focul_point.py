import pyrealsense2 as rs

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

pipeline.start(config)

profile = pipeline.get_active_profile()
depth_stream = profile.get_stream(rs.stream.depth)
intrinsics = depth_stream.as_video_stream_profile().get_intrinsics()

print("fx:", intrinsics.fx)
print("fy:", intrinsics.fy)
print("cx:", intrinsics.ppx)
print("cy:", intrinsics.ppy)