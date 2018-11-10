import cv2
import numpy as np
import glob
import pickle
class LaneDetector:

    def __init__(self):
        self.calculateChessboard()

    #main function that analyzes and find the lanes
    def processLanes(self, img):
        flattenedImage = self.flattenImage(img)
        warpedImage = self.perspective_warp(flattenedImage)
        return warpedImage




    def perspective_warp(self,
                     img, inv = False
                     dst_size=(640,360),
                     src=np.float32([(0.43,0.59),(0.58,0.59),(0,1),(1,1)]),
                     dst=np.float32([(0,0), (1, 0), (0,1), (1,1)])):
        if inv == True:
            dst = tempdst
            dst = src
            src = tempdst
        img_size = np.float32([(img.shape[1],img.shape[0])])
        src = src* img_size
        dst = dst * np.float32(dst_size)
        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(img, M, dst_size)
        return warped



    def calculateChessboard(self):
        obj_pts = np.zeros((6*9,3), np.float32)
        obj_pts[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)
        objpoints = []
        imgpoints = []
        images = glob.glob('../resources/chessboard/*.jpg')
        print(images)
        for indx, fname in enumerate(images):
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            ret, corners = cv2.findChessboardCorners(gray, (9,6), None)

            if ret == True:
                objpoints.append(obj_pts)
                imgpoints.append(corners)
        img_size = (img.shape[1], img.shape[0])
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None,None)
        dst = cv2.undistort(img, mtx, dist, None, mtx)
        dist_pickle = {}
        dist_pickle['mtx'] = mtx
        dist_pickle['dist'] = dist
        self.dump = dist_pickle

    def flattenImage(self,image):
        dir = "../resources/chessboard/final.p"
        mtx = self.dump['mtx']
        dist = self.dump['dist']
        dst = cv2.undistort(image, mtx, dist, None, mtx)
        return dst
