
#import open3d as o3d
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


M = 6
test_images = []

def visualize_projector_mapping(camera_to_projector_mapping, axis='x'):

    coord_index = 0 if axis == 'x' else 1
    projector_coords = camera_to_projector_mapping[:, :, coord_index]

    valid_mask = projector_coords >= 0
    projector_coords[~valid_mask] = -1

    unique_values = np.unique(projector_coords[valid_mask])
    num_unique_values = len(unique_values)


    cmap = plt.cm.gist_rainbow
    norm = mpl.colors.Normalize(vmin=0, vmax=num_unique_values-1)


    value_to_index = {v: i for i, v in enumerate(unique_values)}
    indexed_map = np.full_like(projector_coords, -1)
    for value, index in value_to_index.items():
        indexed_map[projector_coords == value] = index

    indexed_map[~valid_mask] = -1


    plt.figure(figsize=(10, 6))
    plt.title(f"{axis}")
    im = plt.imshow(indexed_map, cmap=cmap, norm=norm)
    plt.axis('off')


    cbar = plt.colorbar(im, orientation='vertical', ticks=np.arange(num_unique_values))
    cbar.ax.set_yticklabels(unique_values)
    cbar.set_label(f"{axis}-coordinate")
    plt.show()




def gray_to_binary(gray_code):
    binary_code = gray_code[0]
    for i in range(1, len(gray_code)):
        binary_code = binary_code + str(int(binary_code[-1]) ^ int(gray_code[i]))
    return int(binary_code, 2)


#old code
def decode_gray(test_images, height, width):

    # height, width = test_images[0].shape

    gray_code = np.zeros((M, height, width), dtype=int)

    for x in range(M-1):
        for col in range(height):
            for row in range(width):

                pixel_value = test_images[x][col, row] / 255.0

                #need a better algorithmn for this as well... this is not ideal
                gray_code[x, col, row] = 0 if pixel_value > 0.5 else 1



    gray_code_sequence = np.empty((height, width), dtype=object)
    binary_sequence = np.zeros((height, width), dtype=int)



    for col in range(height):
        for row in range(width):
            gray_code_sequence[col, row] = ''.join(str(gray_code[x, col, row]) for x in range(M))
            binary_sequence[col, row] = gray_to_binary(gray_code_sequence[col, row])


    #cv2.imshow("Altered", binary_sequence_left.astype(np.uint8))
    #cv2.imshow("Original", binary_sequence.astype(np.uint8))

    return binary_sequence

def decode_gray_otsu(test_images, height, width):
    print("Decoding images...")
    gray_code = np.zeros((len(test_images), height, width), dtype=int)

    for x, img in enumerate(test_images):
        # Calculate threshold using Otsu's method and apply it
        _, thresh_image = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        gray_code[x] = thresh_image / 255  # Normalize to 0 and 1

    gray_code_sequence = np.empty((height, width), dtype=object)
    binary_sequence = np.zeros((height, width), dtype=int)

    for row in range(height):
        for col in range(width):
            gray_code_sequence[row, col] = ''.join(str(gray_code[x, row, col]) for x in range(len(test_images)))
            binary_sequence[row, col] = gray_to_binary(gray_code_sequence[row, col])
    return binary_sequence


def decoding_main():
    for x in range(M*2):
        #filename = f"feb_six/IMG_{x+4577}.JPG"
        filename = f"gray_code_images/IMG_{x+4527}.JPG"
        img = cv2.imread(filename)
        if img is None:
            raise FileNotFoundError(f"Image not found: {filename}")
        img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        height_new, width_new, channel = img.shape

        width_final = 300
        aspect_ratio = height_new/width_new
        height_final = int(width_final*aspect_ratio)


        K = cv2.resize(img_grey, (width_final, height_final))
        test_images.append(K)


    binary_code_hori = decode_gray_otsu(test_images[0:M-1], height_final, width_final)
    binary_code_veri = decode_gray_otsu(test_images[M:-1], height_final, width_final)

    decoded_combine = np.stack((binary_code_hori, binary_code_veri), axis=-1)


    visualize_projector_mapping(decoded_combine, axis = 'x')
    visualize_projector_mapping(decoded_combine, axis='y')


    return binary_code_hori, binary_code_hori

#decoding_main()