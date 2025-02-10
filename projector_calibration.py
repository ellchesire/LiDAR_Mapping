import numpy as np
import cv2
import pickle
from decode_gray import decode_gray_otsu


M = 6
positions = 3
chessboard = (5,5)


def local_homographies(image, corners, decoded, patch_size=47):
    half_size = int(patch_size / 2)
    homographies = []

    for corner in corners:
        x, y = int(corner[0]), int(corner[1])

        # extracting patches
        x1, y1 = max(0, x - half_size), max(0, y - half_size)
        #x2, y2 = min(image.shape[1], x + half_size), min(image.shape[0], y + half_size)
        x2, y2 = min(image.shape[1] - 1, x + half_size), min(image.shape[0] - 1, y + half_size)

        camera_pts = []
        projector_pts = []


        for i in range(y1, y2):
            for j in range(x1, x2):
                    camera_pts.append([i, j])  # camera coordinates
                    projector_pts.append(decoded[i, j])  # projector coordinates
        #  computing homographies
        camera_pts = np.array(camera_pts, dtype=np.float32)
        projector_pts = np.array(projector_pts, dtype=np.float32)

        H, _ = cv2.findHomography(camera_pts, projector_pts, method=cv2.RANSAC)
        homographies.append(H)


    return homographies


def project_img_coords(corners, homographies):

    projector_corners = []

    for i, corner in enumerate(corners):
        H = homographies[i]
        if H is not None:
            x, y = corner
            cam_point = np.array([x, y, 1], dtype=np.float32).reshape(3, 1)
            proj_point = np.dot(H, cam_point)  # apply homography

            #make it not 3D
            proj_x = proj_point[0] / proj_point[2]
            proj_y = proj_point[1] / proj_point[2]
            projector_corners.append((proj_x, proj_y))
        else:
            projector_corners.append(None)

    return projector_corners

#unneeded for now
def proj_calibration(img_corners):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((chessboard[0] * chessboard[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard[0], 0:chessboard[1]].T.reshape(-1, 2)

    objpoints = []  # 3D points in world space
    imgpoints = [np.array(img_corners, dtype=np.float32).reshape(-1, 1, 2)]  # 2D points in image space

    objpoints.append(objp)

    print(f"Number of object points: {len(objpoints)}")
    print(f"Number of image points: {len(imgpoints)}")

    return objpoints, imgpoints




def main():
    objp = np.zeros((chessboard[0] * chessboard[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard[0], 0:chessboard[1]].T.reshape(-1, 2)

    img_org = cv2.imread("cam_calibration/CAM2.JPG")
    height_new, width_new, channel = img_org.shape

    width_final = 300
    aspect_ratio = height_new / width_new
    height_final = int(width_final * aspect_ratio)

    image_groups = np.zeros((positions, M * 2, height_final, width_final), dtype=np.uint8)

    imgpoints_full = []
    imgpoints_pic = []
    objpoints_full = []

    offset = 4605

    for y in range(positions):
        #need to make this for multiple positions
        filename = f"cam_calibration\CAM{y+1}.JPG"
        img_corner = cv2.imread(filename)

        gray = cv2.cvtColor(img_corner, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (width_final, height_final))
        ret, corners = cv2.findChessboardCorners(gray, (chessboard[0], chessboard[1]), None)
        corners_squeezed = corners.squeeze()

        for x in range(M * 2):
            #filename = f"gray_code_images/NORMAL{x + 50:05d}.JPG"
            print(y, offset)
            filename = f"proj_calibration\IMG_{offset}.JPG"
            img = cv2.imread(filename)
            img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            K = cv2.resize(img_grey, (width_final, height_final))
            image_groups[y,x] = K

            offset+=1


        binary_code_hori = decode_gray_otsu(image_groups[y,0:M-1], height_final, width_final)
        binary_code_veri = decode_gray_otsu(image_groups[y,M:-1], height_final, width_final)

        # cv2.imshow("veri", binary_code_veri.astype(np.uint8))
        # cv2.imshow("hori", binary_code_hori.astype(np.uint8))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        decoded_combine = np.stack((binary_code_hori, binary_code_veri), axis=-1)


        homog = local_homographies(gray, corners_squeezed, decoded_combine)
        project_coords = project_img_coords(corners_squeezed, homog)
        
        
        objpoints_full.append(objp)
        imgpoints_full.append(np.array(project_coords, dtype=np.float32).reshape(-1, 1, 2))
        
        if(y==1):
            imgpoints_pic.append(np.array(project_coords, dtype=np.float32).reshape(-1, 1, 2))
            f = open('proj_img', 'wb')
            pickle.dump(imgpoints_pic, f)
            f.close()


    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints_full, imgpoints_full, gray.shape[::-1], None, None)
    print(ret)

    f = open('projector_calibration', 'wb')
    pickle.dump(mtx, f)
    pickle.dump(dist, f)
    f.close()



if __name__ == '__main__':
    main()