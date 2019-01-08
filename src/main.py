import numpy as np
import cv2
from LaneDetector import LaneDetector
from newLaneDetector import NewLaneDetector

#Configs

videoConfig = "HardTrimmed"
output = "..\\resources\\output_videos\\first.avi"
def main():
    input,dimensions = config(videoConfig)
    laneDetector = NewLaneDetector(dimensions)
    if output != "":
        video = cv2.VideoWriter(output,cv2.VideoWriter_fourcc(*'DIVX'),30,(1280,720))

    cap = cv2.VideoCapture(input)
    while(1):
        ret, frame = cap.read()
        if ret == False:
            break
        frame = laneDetector.processLanes(frame)
        if output != "":
            video.write(frame)

        #frames.append(frame)
        cv2.imshow('frame',frame)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    #saveToVideo(frames,"..\\resources\\output_videos\\first.avi")
    if output != "":
        video.release()
    cap.release()
    cv2.destroyAllWindows()


def config(name):
    if name == "HardTrimmed":
        input = '..\\resources\\stockcar\\hard.mp4'
        dimensions = [5/100,1,47/100,70/100,53/100,70/100,95/100,1]
    return input,dimensions

if __name__ == "__main__":
    main()
