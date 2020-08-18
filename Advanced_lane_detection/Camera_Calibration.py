import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
"""
    A list of images are read to get list of their object points which will be the same for all the images since it's 
    the same chess board, and image points which are found using the cv2.findchessboardcorners function. A list of these
    object and image points taken from 20 images when sent into the cv2.calibratecamera function gives us the calibration
    matrix. This matrix can be universally used to get destination image.
"""
image = glob.glob('../SelfDriving_Lesson2/Calibration images/GOPR00*.jpg')
objpoints = []
imgpoints = []

for frame in image:

    img = mpimg.imread(frame)

    objp = np.zeros((8*6, 3), np.float32)
    objp[:, :2] = np.mgrid[0:8, 0:6].T.reshape(-1, 2)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (8, 6), None)

    if ret:
        imgpoints.append(corners)
        objpoints.append(objp)

img = mpimg.imread('calibration_test1.jpg')
plt.imshow(img)
plt.show()
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[1::-1], None, None)
dst = cv2.undistort(img, mtx, dist, None, mtx)
plt.imshow(dst)
plt.show()
