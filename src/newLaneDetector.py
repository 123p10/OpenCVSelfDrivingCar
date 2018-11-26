import cv2
import numpy as np
from matplotlib import pyplot as plt

class NewLaneDetector:
    def processLanes(self,frame):
        transformedImage = self.transformLane(frame)
        cannyedImage = self.cannyImage(transformedImage)
        return cannyedImage
    def transformLane(self,frame):
        perspectiveTransform = cv2.getPerspectiveTransform(np.float32([[0,560],[0,720],[1280,720],[1280,560]]),np.float32([[0,0],[0,720],[1280,720],[1280,0]]))
        out = cv2.warpPerspective(frame,perspectiveTransform,(1280,720))
        return out
    def cannyImage(self,frame):
        return cv2.Canny(frame,100,150)
