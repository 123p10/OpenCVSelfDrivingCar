import numpy as np
import cv2
from LaneDetector import LaneDetector
from newLaneDetector import NewLaneDetector
from carDetector import CarDetector
from yoloCarDetector import YoloCarDetector

#Configs
videoConfig = "BrownCar"
output = "..\\resources\\output_videos\\BrownCar.avi"
classifierPath = "..\\resources\\xml_files\\cars.xml"
def main():
    input,dimensions,white_filter,yellow_filter = config(videoConfig)
    laneDetector = NewLaneDetector(dimensions,yellow_filter=yellow_filter,white_filter=white_filter)
    #carDetector = CarDetector(classifierPath)
    yoloCarDetector = YoloCarDetector("..\\resources\\car_detector\\yolo3.cfg","..\\resources\\car_detector\\yolov3.weights","..\\resources\\car_detector\\yolo.txt")
    if output != "":
        video = cv2.VideoWriter(output,cv2.VideoWriter_fourcc(*'DIVX'),30,(1280,720))

    cap = cv2.VideoCapture(input)
    while(1):
        ret, frame = cap.read()
        if ret == False:
            break
        frame = laneDetector.processLanes(frame)
        #remember to uncomment this
        #frame = carDetector.detectCars(frame)
        frame = yoloCarDetector.detectCars(frame)
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
        dimensions = [5/100,1,47/100,75/100,53/100,75/100,95/100,1]
        yellow_filter = [15, 38, 115]
        white_filter = [0, 180, 0]
    if name == "BrownCar":
        input = '..\\resources\\stockcar\\browncar.mp4'
        dimensions = [10/100,11/12,47/100,63/100,53/100,63/100,90/100,11/12]
        yellow_filter = [15, 38, 115]
        white_filter = [0, 200, 0]
    return input,dimensions,white_filter,yellow_filter

if __name__ == "__main__":
    main()
