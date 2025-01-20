import numpy as np
import cv2 as cv

M = 13
chess = (6,9)

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((chess[0]*chess[1],3), np.float32)
objp[:,:2] = np.mgrid[0:chess[0],0:chess[1]].T.reshape(-1,2)

objpoints = []
imgpoints = []

for x in range(M):
    filename = f"checker{x + 1 }.JPG"
    #1, 11

    img = cv.imread(filename)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)


    # cv.imshow("Gray Image", img_gray)
    #
    # cv.waitKey(0)
    # Find the chess board corners

    # test_image = cv.imread('img.png')
    # gray = cv.cvtColor(test_image, cv.COLOR_BGR2GRAY)
    # ret, corners = cv.findChessboardCorners(gray, (6, 9), None)
    # print("Test Chessboard detection result:", ret)

    ret, corners = cv.findChessboardCorners(gray, (chess[0], chess[1]), None)

    # If found, add object points, image points (after refining them)
    print("Test Chessboard detection result:", x+1, ret)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        cv.drawChessboardCorners(img, (chess[0], chess[1]), corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(0)


ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
print(mtx)