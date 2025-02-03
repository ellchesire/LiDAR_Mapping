import numpy as np
import cv2
import pickle
from decode_gray import  decoding_main

chessboard = (5,5)

def triangulate_points(horizontal_indices, vertical_indices, cam_mtx, proj_mtx, R, T):
    height, width = horizontal_indices.shape


    P_cam = cam_mtx @ np.hstack((np.eye(3), np.zeros((3, 1))))  # Camera matrix
    P_proj = proj_mtx @ np.hstack((R, T.reshape(-1, 1)))       # Projector matrix

    points_3D = []
    for y in range(height):
        for x in range(width):

            u_proj = horizontal_indices[y, x]
            v_proj = vertical_indices[y, x]

            u_cam, v_cam = x, y

            cam_point = np.array([u_cam, v_cam, 1.0])
            proj_point = np.array([u_proj, v_proj, 1.0])

            point_3D_homo = cv2.triangulatePoints(P_cam, P_proj, cam_point[:2], proj_point[:2])
            point_3D = point_3D_homo[:3] / point_3D_homo[3]
            points_3D.append(point_3D)

    return np.array(points_3D).reshape((height, width, 3))

def main():
    objp = np.zeros((chessboard[0] * chessboard[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard[0], 0:chessboard[1]].T.reshape(-1, 2)

    objpoints = []
    objpoints.append(objp)

    f = open('camera_calibration', 'rb')
    cam_mtx = pickle.load(f)
    cam_dist = pickle.load(f)
    cam_imgpoints = pickle.load(f)

    f.close()

    f = open('projector_calibration', 'rb')
    proj_mtx = pickle.load(f)
    proj_dist = pickle.load(f)
    proj_imgpoints = pickle.load(f)

    f.close()

    filename = "IMG_4458.JPG"
    img_corner = cv2.imread(filename)

    height_new, width_new, channel = img_corner.shape

    width_final = 300
    aspect_ratio = height_new / width_new
    height_final = int(width_final * aspect_ratio)

    gray = cv2.cvtColor(img_corner, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (width_final, height_final))


    ret, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
        objpoints,  # 3D object points
        cam_imgpoints,  # 2D points in camera
        proj_imgpoints,  # 2D points in proj
        cam_mtx,  # camera matrix
        cam_dist,  #  camera distortion coefficients
        proj_mtx,  # proj matrix
        proj_dist,  # proj distortion coefficients
        gray.shape[::-1],  # Image size (width, height)
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),  # Termination criteria
        flags=0
    )

    hori, veri = decoding_main()

    triangulate_points(hori, veri, cam_mtx, proj_mtx, R, T)



if __name__ == '__main__':
    main()