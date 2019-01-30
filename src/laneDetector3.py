import cv2
import numpy as np
from matplotlib import pyplot as plt

class LaneDetector3:
    dimensions = []
    yellow_filter = []
    white_filter = []
    L = [1280,0]
    R = [720,0]
    def __init__(self,dim,yellow_filter,white_filter):
        self.yellow_filter = yellow_filter
        self.white_filter = white_filter
        self.dimensions = dim
    def processLanes(self,frame):
        self.L = [1280,0]
        self.R = [720,0]
        hsled_img = self.filter_img_hsl(frame);
        greyscale = cv2.cvtColor(hsled_img,cv2.COLOR_BGR2GRAY)
        #blur = cv2.GaussianBlur(greyscale, (5, 5), 0)
        blur = cv2.medianBlur(greyscale,5)
        cannyedImage = self.cannyImage(blur)
        croppedImage = self.region_of_interest(cannyedImage)
        kernel = np.ones((3,3),np.uint8)
        dilated = cv2.dilate(croppedImage,kernel,iterations = 1)
        lines = self.hough_lines(dilated)
        lLane,rLane = self.separate_lines(lines,frame)
        rightfit = self.getPoly(rLane,'R')

        leftfit = self.getPoly(lLane,'L')
        colored = self.newDrawLanes(leftfit,rightfit,frame)
        #colored = self.shadeLanes(leftfit,rightfit,frame)
        return colored

    def shadeLanes(self,leftLane,rightLane,img):
        if len(leftLane) == 3:
            ptL = leftLane[0] * self.L[1]*self.L[1] + leftLane[1] * self.L[1] + leftLane[2]
        else:
            ptL = leftLane[0] * self.L[1] + leftLane[1]
        if len(rightLane) == 3:
            ptR = rightLane[0] * self.R[0]*self.L[1] + rightLane[1] * self.R[0] + rightLane[2]
        else:
            ptR = rightLane[0] * self.R[0] + rightLane[1]

        if ptL < ptR:
            max = ptL
        else:
            max = ptR


        for i in range(0,1280):
            if len(leftLane) == 3:
                pt1 = leftLane[0] * i*i + leftLane[1] * i + leftLane[2]
                pt2 = leftLane[0] * (i+5)*(i+5) + leftLane[1] * (i+5) + leftLane[2]
            else:
                pt1 = leftLane[0] * i + leftLane[1]
                pt2 = leftLane[0] *(i+5) + leftLane[1]
            if pt2 >= max and i > self.L[0]:
                cv2.line(img,(i,int(pt1)),(i+5,int(pt2)),(255,0,0),5)

            if len(rightLane) == 3:
                pt1 = rightLane[0] * i*i + rightLane[1] * i + rightLane[2]
                pt2 = rightLane[0] * (i+5)*(i+5) + rightLane[1] * (i+5) + rightLane[2]
            else:
                pt1 = rightLane[0] * i + rightLane[1]
                pt2 = rightLane[0] * (i+5) + rightLane[1]
            if pt1 >= max and i < self.R[1]:
                cv2.line(img,(i,int(pt1)),(i+5,int(pt2)),(0,255,0),5)
            i += 5
        return img
    def newDrawLanes(self,leftLane,rightLane,img):
        for i in range(0,1280):
            if i < self.L[1] and i > self.L[0]:
                if len(leftLane) == 3:
                    pt1 = leftLane[0] * i*i + leftLane[1] * i + leftLane[2]
                    pt2 = leftLane[0] * (i+5)*(i+5) + leftLane[1] * (i+5) + leftLane[2]
                else:
                    pt1 = leftLane[0] * i + leftLane[1]
                    pt2 = leftLane[0] *(i+5) + leftLane[1]

                cv2.line(img,(i,int(pt1)),(i+5,int(pt2)),(255,0,0),5)

            if i < self.R[1] and i > self.R[0]:
                if len(rightLane) == 3:
                    pt1 = rightLane[0] * i*i + rightLane[1] * i + rightLane[2]
                    pt2 = rightLane[0] * (i+5)*(i+5) + rightLane[1] * (i+5) + rightLane[2]
                else:
                    pt1 = rightLane[0] * i + rightLane[1]
                    pt2 = rightLane[0] * (i+5) + rightLane[1]
                cv2.line(img,(i,int(pt1)),(i+5,int(pt2)),(0,255,0),5)
            i += 5
        return img
    def getPoly(self,lines,side):
        x = []
        y = []
        if len(lines) == 0:
            return 0

        for line in lines:
            #print(line[0][0])
            if side == 'L':
                if self.L[0] > line[0][0]:
                    self.L[0] = line[0][0]
                if self.L[1] < line[0][2]:
                    self.L[1] = line[0][2]
            else:
                if self.R[0] > line[0][0]:
                    self.R[0] = line[0][0]
                if self.R[1] < line[0][2]:
                    self.R[1] = line[0][2]

            x.append(line[0][0])
            y.append(line[0][1])
            x.append(line[0][2])
            y.append(line[0][3])
        fit = np.polyfit(x,y,2)
        #print(fit[0])
        if abs(fit[0]) >= 0.0005:
            #print("\n\n\\naaaa")
            fit = np.polyfit(x,y,1)


        return fit;
    def color_lanes(self,img, left_lane_lines, right_lane_lines, left_lane_color=[255, 0, 0], right_lane_color=[0, 0, 255]):
        left_colored_img = self.draw_lines(img, left_lane_lines, color=left_lane_color, make_copy=True)
        right_colored_img = self.draw_lines(left_colored_img, right_lane_lines, color=right_lane_color, make_copy=False)
        return right_colored_img
    def separate_lines(self,lines, img):
        img_shape = img.shape
        middle_x = img_shape[1] / 2
        left_lane_lines = []
        right_lane_lines = []
        for line in lines:
            for x1, y1, x2, y2 in line:
                dx = x2 - x1
                if dx == 0:
                    #Discarding line since we can't gradient is undefined at this dx
                    continue
                dy = y2 - y1

                # Similarly, if the y value remains constant as x increases, discard line
                if dy == 0:
                    continue

                slope = dy / dx
                # This is pure guess than anything...
                # but get rid of lines with a small slope as they are likely to be horizontal one
                epsilon = 0.1
                if abs(slope) <= epsilon:
                    continue

                if slope < 0 and x1 < middle_x and x2 < middle_x:
                    # Lane should also be within the left hand side of region of interest
                    left_lane_lines.append([[x1, y1, x2, y2]])
                elif x1 >= middle_x and x2 >= middle_x:
                    # Lane should also be within the right hand side of region of interest
                    right_lane_lines.append([[x1, y1, x2, y2]])

        return left_lane_lines, right_lane_lines
    def hough_lines(self,img):
        rho = 1
        theta = (np.pi/180) * 1
        threshold = 15
        min_line_len = 15
        max_line_gap = 10
        return cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    def draw_lines(self,img, lines, color=[255, 0, 0], thickness=10, make_copy=True):
        # Copy the passed image
        img_copy = np.copy(img) if make_copy else img

        for line in lines:
            for x1,y1,x2,y2 in line:
                cv2.line(img_copy, (x1, y1), (x2, y2), color, thickness)

        return img_copy
    def region_of_interest(self,img):
        #defining a blank mask to start with
        mask = np.zeros_like(img)

        #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
        if len(img.shape) > 2:
            channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255

        vert = np.array([[(img.shape[1]*self.dimensions[0],img.shape[0]*self.dimensions[1]) , (img.shape[1]*self.dimensions[2],img.shape[0]*self.dimensions[3]), (img.shape[1]*self.dimensions[4],img.shape[0]*self.dimensions[5]), (img.shape[1]*self.dimensions[6],img.shape[0]*self.dimensions[7])]], dtype=np.int32)

        #filling pixels inside the polygon defined by "vertices" with the fill color
        cv2.fillPoly(mask, vert, ignore_mask_color)

        #returning the image only where mask pixels are nonzero
        masked_image = cv2.bitwise_and(img, mask)
        return masked_image
    #Unused
    def transformLane(self,frame):
        perspectiveTransform = cv2.getPerspectiveTransform(np.float32([[0,650],[0,720],[1280,720],[1280,650]]),np.float32([[0,0],[0,720],[1280,720],[1280,0]]))
        out = cv2.warpPerspective(frame,perspectiveTransform,(1280,720))
        return out
    def cannyImage(self,frame):
        return cv2.Canny(frame,75,100)


    def combine_hsl_isolated_with_original(self,img, hsl_yellow, hsl_white):
        hsl_mask = cv2.bitwise_or(hsl_yellow, hsl_white)
        return cv2.bitwise_and(img, img, mask=hsl_mask)

    def filter_img_hsl(self,img):
        hsl_img = cv2.cvtColor(img,cv2.COLOR_BGR2HLS)
        hsl_yellow = self.isolate_yellow_hsl(hsl_img)
        hsl_white = self.isolate_white_hsl(hsl_img)
        return self.combine_hsl_isolated_with_original(img, hsl_yellow, hsl_white)
    def isolate_yellow_hsl(self,img):
        # Caution - OpenCV encodes the data in ****HLS*** format
        # Lower value equivalent pure HSL is (30, 45, 15)
        low_threshold = np.array(self.yellow_filter, dtype=np.uint8)
        # Higher value equivalent pure HSL is (75, 100, 80)
        high_threshold = np.array([35, 204, 255], dtype=np.uint8)

        yellow_mask = cv2.inRange(img, low_threshold, high_threshold)

        return yellow_mask


    # Image should have already been converted to HSL color space
    def isolate_white_hsl(self,img):
        # Caution - OpenCV encodes the data in ***HLS*** format
        # Lower value equivalent pure HSL is (30, 45, 15)
        low_threshold = np.array(self.white_filter, dtype=np.uint8)
        # Higher value equivalent pure HSL is (360, 100, 100)
        high_threshold = np.array([180, 255, 255], dtype=np.uint8)

        yellow_mask = cv2.inRange(img, low_threshold, high_threshold)

        return yellow_mask
