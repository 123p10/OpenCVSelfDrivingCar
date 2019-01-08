import numpy as np
import cv2
from LaneDetector import LaneDetector
from newLaneDetector import NewLaneDetector

#frames = []
def main():
    laneDetector = NewLaneDetector()
    video = cv2.VideoWriter("..\\resources\\output_videos\\first.avi",cv2.VideoWriter_fourcc(*'DIVX'),30,(1280,720))

    cap = cv2.VideoCapture('..\\resources\\stockcar\\trimmed.mp4')
    while(1):
        ret, frame = cap.read()
        if ret == False:
            break
        frame = laneDetector.processLanes(frame)
        output = frame
        video.write(frame)

        #frames.append(frame)
        cv2.imshow('frame',output)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    #saveToVideo(frames,"..\\resources\\output_videos\\first.avi")
    video.release()
    cap.release()
    cv2.destroyAllWindows()

def saveToVideo(imgs,path):
    video = cv2.VideoWriter(path,cv2.VideoWriter_fourcc(*'DIVX'),30,(1280,720))
    for i in range(len(imgs)):
        video.write(imgs[i])
    video.release()

if __name__ == "__main__":
    main()
