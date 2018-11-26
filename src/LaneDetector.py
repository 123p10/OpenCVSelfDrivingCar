import cv2
import numpy as np
import glob
import pickle
import math

class LaneDetector:

    def __init__(self):
        return

    def processLanes(self, img):
        region_of_interest_vertices = [
            (0, 720),
            (1280/2, 720*(21/30)),
            (1280, 720),
        ]
        sizedImage = cv2.resize(img,(1280,720))
        greyScaledImage = cv2.cvtColor(sizedImage,cv2.COLOR_BGR2GRAY)
        gaussianBlur = cv2.GaussianBlur(greyScaledImage,(5,5),0)
        cannyedImage = self.cannyImage(gaussianBlur,25,100)
        croppedImage = self.cropImage(cannyedImage,np.array([region_of_interest_vertices],np.int32),)
        lines = cv2.HoughLinesP(
            croppedImage,
            rho = 6,
            theta = np.pi / 60,
            threshold = 160,
            lines = np.array([]),
            minLineLength = 40,
            maxLineGap = 25
        )
        output = self.draw_lines(sizedImage,lines)
        return output

    def cannyImage(self,img,min,max):
        return cv2.Canny(img,min,max)

    def cropImage(self, img, vertices):
        mask = np.zeros_like(img)
        channel_count = 1
        match_mask_color = (255,) * channel_count
        cv2.fillPoly(mask, np.int32(vertices), match_mask_color)
        masked_image = cv2.bitwise_and(img, mask)
        return masked_image

    def draw_lines(self,img, lines, color=[255, 0, 0], thickness=3):
        if lines is None:
            return
        img = np.copy(img)
        line_img = np.zeros(
            (
                img.shape[0],
                img.shape[1],
                3
            ),
            dtype=np.uint8,
        )
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_img, (x1, y1), (x2, y2), color, thickness)
        img = cv2.addWeighted(img, 0.8, line_img, 1.0, 0.0)
        return img
