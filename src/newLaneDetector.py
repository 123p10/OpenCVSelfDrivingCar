import cv2
import numpy as np
from matplotlib import pyplot as plt

class NewLaneDetector:
    def processLanes(self,frame):
        transformedImage = self.transformLane(frame)
        cannyedImage = self.cannyImage(transformedImage)
        return cannyedImage
    def transformLane(self,frame):
        perspectiveTransform = cv2.getPerspectiveTransform(np.float32([[0,650],[0,720],[1280,720],[1280,650]]),np.float32([[0,0],[0,720],[1280,720],[1280,0]]))
        out = cv2.warpPerspective(frame,perspectiveTransform,(1280,720))
        return out
    def cannyImage(self,frame):
        return cv2.Canny(frame,100,150)
    def calcLocation():

        return 0
    def calcHist(self,frame):
        hist = cv2.calcHist([frame],[0],None,[256],[0,256])
        a = 0
        for i in hist:
            if i > 10:
                cv2.line(frame,(a,0),(a,700),(255,0,0),5)
                print(i)
            a += 1
        return frame
