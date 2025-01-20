import numpy as np
import cv2 as cv

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((3*7,3), np.float32)
objp[:,:2] = np.mgrid[0:3,0:7].T.reshape(-1,2)

objpoints = []
imgpoints = []

image = 'NORMAL00042.jpg'

#for fname in image:

img = cv.imread(image)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
img_gray = cv.resize(gray, (100,200))
# cv.imshow("Gray Image", img_gray)
#
# cv.waitKey(0)
# Find the chess board corners

test_image = cv.imread('img.png')
gray = cv.cvtColor(test_image, cv.COLOR_BGR2GRAY)
ret, corners = cv.findChessboardCorners(gray, (6, 9), None)
print("Test Chessboard detection result:", ret)


ret, corners = cv.findChessboardCorners(gray, (3, 7), None)


# If found, add object points, image points (after refining them)
print("Test Chessboard detection result:", ret)
if ret == True:
    objpoints.append(objp)

    corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    imgpoints.append(corners2)

    # Draw and display the corners
    cv.drawChessboardCorners(img, (3, 7), corners2, ret)
    cv.imshow('img', img)
    print("i have reached here")
    cv.waitKey(0)

#doesn't work yet
# ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)