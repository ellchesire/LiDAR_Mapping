

import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
from sympy.physics.quantum.gate import normalized

M = 6
test_images = []

#need to figure out the aspect ratio
# height = 300
# width = 300

#disparity parameters
disparity_range = [0, 64]
block_size = 5
# Load and process images
# for x in range(M):
#     filename = f"test_image_{x+1}.PNG"
#     img = cv2.imread(filename)
#     img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     K = cv2.resize(img_grey, (150, 100))
#     test_images.append(K)

# get size of a new image

# relook at this
def gray_to_binary(gray_code):
    binary_code = gray_code[0]
    for i in range(1, len(gray_code)):
        binary_code = binary_code + str(int(binary_code[-1]) ^ int(gray_code[i]))
    return int(binary_code, 2)



def decode_gray(test_images, height, width):

    # height, width = test_images[0].shape

    gray_code = np.zeros((M, height, width), dtype=int)


    for x in range(M):
        for col in range(height):
            for row in range(width):

                pixel_value = test_images[x][col, row] / 255.0

                #need a better algorithmn for this as well... this is not ideal
                gray_code[x, col, row] = 0 if pixel_value > 0.5 else 1


    gray_code_sequence = np.empty((height, width), dtype=object)
    binary_sequence = np.zeros((height, width), dtype=int)
    binary_sequence_left = np.zeros((height, width), dtype=int)


    for col in range(height):
        for row in range(width):
            gray_code_sequence[col, row] = ''.join(str(gray_code[x, col, row]) for x in range(M))
            binary_sequence[col, row] = gray_to_binary(gray_code_sequence[col, row])
            # if random.random() > 0.3:
            #     binary_sequence_left[col, row] = binary_sequence[col, row] + 50

    # cv2.imshow("Altered", binary_sequence_left.astype(np.uint8))
    # cv2.imshow("Original", binary_sequence.astype(np.uint8))


    image = binary_sequence.astype(np.uint8)

    return image
#

#old disparity map code
# def disparityMap(left_image, right_image):
#
#
#     stereo = cv2.StereoBM_create(numDisparities=disparity_range[1], blockSize=block_size)
#     disparity_map = stereo.compute(left_image, right_image)
#
#     # Display disparity map
#     plt.figure()
#     plt.imshow(disparity_map, cmap='jet')
#     plt.colorbar()
#     plt.title('Disparity Map')
#     plt.show()


def main():
    for x in range(M):
        filename = f"IMG_{x + 4360 }.JPG"
        img = cv2.imread(filename)
        if img is None:
            raise FileNotFoundError(f"Image not found: {filename}")
        img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        height_new, width_new, channel = img.shape

        width_final = 300
        aspect_ratio = height_new/width_new
        height_final = int(width_final*aspect_ratio)

        print(height_final)

        #need to find a correct sizing (i think the actual one is 2250, 22500 or something like that, insane
        K = cv2.resize(img_grey, (width_final, height_final))
        test_images.append(K)




    cv2.imshow("test", test_images[0])
    cv2.imshow("test1", test_images[1])
    cv2.imshow("test2", test_images[2])
    cv2.imshow("test3", test_images[3])
    cv2.imshow("test4", test_images[4])
    cv2.imshow("test5", test_images[5])


    binary_code = decode_gray(test_images, height_final, width_final)

    normalized_gray = (binary_code / binary_code.max() * 255).astype(np.uint8)

    cv2.imshow("decoded", normalized_gray)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
