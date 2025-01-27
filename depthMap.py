import numpy as np
import cv2 as cv

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

            point_3D_homo = cv.triangulatePoints(P_cam, P_proj, cam_point[:2], proj_point[:2])
            point_3D = point_3D_homo[:3] / point_3D_homo[3]
            points_3D.append(point_3D)

    return np.array(points_3D).reshape((height, width, 3))