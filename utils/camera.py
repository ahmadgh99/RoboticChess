import pyrealsense2 as rs
import numpy as np


class Camera:

    def __init__(self):
        self.frame = None
        self.pipeline = None

    def config_cam(self):
        # Configuration for the Realsense camera based on boardDetection3.py
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        pipeline.start(config)
        self.pipeline = pipeline

    def capture_frame(self):
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            return None
        self.frame = np.asanyarray(color_frame.get_data())
        return np.asanyarray(color_frame.get_data())

    def capture_depth(self):
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        if not depth_frame:
            return None
        return np.asanyarray(depth_frame.get_data())
        
    def quit(self):
        self.pipeline.stop()
