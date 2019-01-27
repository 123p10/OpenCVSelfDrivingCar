import cv2

class CarDetector:
    classifierPath = ""
    def __init__(self,classifier):
            self.classifierPath = classifier

    def detectCars(self,frame):
        greyscale = self.convertToGreyScale(frame)
        classifier = self.loadClassifier(self.classifierPath)
        cars = classifier.detectMultiScale(greyscale, 1.1, 1)
        output = self.processCars(frame,cars)
        return output

    def loadClassifier(self,classifier):
        classifier = cv2.CascadeClassifier(self.classifierPath)
        return classifier

    def convertToGreyScale(self,image):
        image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        return image

    def processCars(self,carFrame,carClass):
        for (x,y,w,h) in carClass:
            cv2.rectangle(carFrame,(x,y),(x+w,y+h),(0,0,255),2)
        return carFrame
