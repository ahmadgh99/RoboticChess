import cv2
import numpy as np


class CustomError(Exception):
    pass

class motion_detector:

    def __init__(self):      
        self.background = None
        self.factor = 15
        self.threshold = 307200/15

    def get_background(self,depth_frame):
        self.background = depth_frame

    def set_threshold(self,factor):
        self.factor = factor
        if factor == 0:
            self.threshold = 0
            return
        self.threshold = 307200/factor
    def get_depth_factor(self):
        return self.factor


    def detect_motion(self,depth_frame):
    
        if self.background is None:
            raise CustomError("Please use get_background first")

        # Compute the difference
        diff = cv2.absdiff(self.background, depth_frame)
        _, thresholded = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

        # Noise reduction
        kernel = np.ones((5,5), np.uint8)
        thresholded = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel)

        total_motion = np.sum(thresholded) / 255

        if total_motion > self.threshold:
            return True
        
        return False

