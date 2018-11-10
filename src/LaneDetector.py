import cv2
import numpy as np
class LaneDetector:
    #main function that analyzes and find the lanes
    def processLanes(self, img):
        greyScale = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        canny = cv2.Canny(greyScale, 100, 200)
        output = self.cropLanes(canny)
    #    output = self.calculateLines(croppedImage)
        return output

    def calculateLines(self,img):
        lines = cv2.HoughLinesP(
            img,
            rho=6,
            theta=np.pi / 60,
            threshold=160,
            lines=np.array([]),
            minLineLength=40,
            maxLineGap=25
        )
        return self.draw_lines(img,lines)
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
        img = cv2.addWeighted(img, 0.8, line_image, 1.0, 0.0)
        return img



    #Fine the region and crop the image
    def cropLanes(self,image):
        height, width = image.shape
        region_of_interest_vertices = [
            (-200, height),
            (width / 2, height / 3),
            (width+200, height),
        ]
        return self.region_of_interest(
            image,
            np.array([region_of_interest_vertices], np.int32),
        )

    #Do the actual cropping
    def region_of_interest(self,img, vertices):
        mask = np.zeros_like(img)
        channel_count = 1
        match_mask_color = (255,) * channel_count
        cv2.fillPoly(mask, vertices, match_mask_color)
        masked_image = cv2.bitwise_and(img, mask)
        return masked_image
