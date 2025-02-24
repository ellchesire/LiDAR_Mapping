import numpy as np
import cv2
import pickle
#TO DO: HAVE THE ACTUAL SIZE OF THE CHECKERBOARD SQUARES


def calibration():
    M = 4
    chess = (5,5)

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objp = np.zeros((chess[0]*chess[1],3), np.float32)
    objp[:,:2] = np.mgrid[0:chess[0],0:chess[1]].T.reshape(-1,2)


    objpoints = []
    imgpoints = []

    for x in range(M):
        #filename = f"IMG_{x+4555}.JPG"
        #filename = f"cam_calibration\IMG_{x+4604}.jpg"
        filename = f"cam_calibration\CAM{x+1}.JPG"
        #1, 11

        img = cv2.imread(filename)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, (chess[0], chess[1]), None)
        
        # If found, add object points, image points (after refining them)
        print("Test Chessboard detection result:", x+1, ret)
        if ret == True:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            print(corners2.shape)
            imgpoints.append(corners2)
            # cv2.drawChessboardCorners(img, (chess[0], chess[1]), corners2, ret)
            # cv2.namedWindow("output", cv2.WINDOW_NORMAL)
            # cv2.imshow("output", img)
            # cv2.waitKey(0)
            #Draw and display the corners
            #

    if(x == 1):
        f = open('cam_img', 'wb')
        pickle.dump(imgpoints, f)
        f.close()

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    print(ret)

    f= open('camera_calibration', 'wb')
    pickle.dump(mtx, f)
    pickle.dump(dist, f)
    f.close()

calibration()