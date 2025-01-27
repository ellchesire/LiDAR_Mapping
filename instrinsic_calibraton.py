import numpy as np
import cv2 as cv
#TO DO: HAVE THE ACTUAL SIZE OF THE CHECKERBOARD SQUARES


def calibration():
    M = 9
    chess = (5,5)

    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objp = np.zeros((chess[0]*chess[1],3), np.float32)
    objp[:,:2] = np.mgrid[0:chess[0],0:chess[1]].T.reshape(-1,2)

    objpoints = []
    imgpoints = []

    for x in range(M):
        filename = f"IMG_{x+4555}.JPG"
        #1, 11

        img = cv.imread(filename)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        ret, corners = cv.findChessboardCorners(gray, (chess[0], chess[1]), None)

        # If found, add object points, image points (after refining them)
        print("Test Chessboard detection result:", x+1, ret)
        if ret == True:
            objpoints.append(objp)

            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            cv.drawChessboardCorners(img, (chess[0], chess[1]), corners2, ret)
            # cv.namedWindow("output", cv.WINDOW_NORMAL)
            # cv.imshow("output", img)
            # cv.waitKey(0)
            # Draw and display the corners
            #


    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    print(ret)

    return mtx


calibration()