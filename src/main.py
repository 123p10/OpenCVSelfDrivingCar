import numpy as np
import cv2
from LaneDetector import LaneDetector
from newLaneDetector import NewLaneDetector

def main():
    laneDetector = NewLaneDetector()
    cap = cv2.VideoCapture('..\\resources\\stockcar\\trimmed.mp4')
    #fgbg = cv2.createBackgroundSubtractorMOG2()
    while(1):
        ret, frame = cap.read()
        frame = laneDetector.processLanes(frame)
        output = frame
        cv2.imshow('frame',output)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()




if __name__ == "__main__":
    main()
