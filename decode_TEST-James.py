import cv2
import numpy as np
import matplotlib.pyplot as plt

def gray_to_binary(gray_code):
    binary_code = gray_code[0]
    for i in range(1, len(gray_code)):
        binary_code = binary_code + str(int(binary_code[-1]) ^ int(gray_code[i]))
    return int(binary_code, 2)

def decode_gray(test_images, height, width):
    print("Decoding images...")  
    gray_code = np.zeros((len(test_images), height, width), dtype=int)

    for x, img in enumerate(test_images):
        # Calculate threshold using Otsu's method and apply it
        _, thresh_image = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        gray_code[x] = thresh_image / 255  # Normalize to 0 and 1

    gray_code_sequence = np.empty((height, width), dtype=object)
    binary_sequence = np.zeros((height, width), dtype=int)

    for col in range(height):
        for row in range(width):
            gray_code_sequence[col, row] = ''.join(str(gray_code[x, col, row]) for x in range(len(test_images)))
            binary_sequence[col, row] = gray_to_binary(gray_code_sequence[col, row])
    cv2.imshow("Original", binary_sequence.astype(np.uint8))
    return binary_sequence

def main():
    test_images = []
    print("Loading images...")  # Debugging
    for x in range(6):
        filename = f"gray_code_images/NORMAL{x + 50:05d}.JPG"
        img = cv2.imread(filename)
        if img is None:
            raise FileNotFoundError(f"Image not found: {filename}")
        print(f"Loaded {filename}")  # Debugging statement
        img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        test_images.append(img_grey)

    if not test_images:
        print("No images loaded")
        return

    height, width = test_images[0].shape

    binary_code = decode_gray(test_images, height, width)
    print("Images processed, displaying results...")

    #cv2.imshow("test", test_images[0])
    #cv2.imshow("test1", test_images[1])
    #cv2.imshow("test2", test_images[2])
    #cv2.imshow("test3", test_images[3])
    #cv2.imshow("test4", test_images[4])
    #cv2.imshow("test5", test_images[5])

    plt.figure(figsize=(8, 8))
    plt.imshow(binary_code, cmap='gray', interpolation='nearest')
    plt.title("Decoded Layered")
    plt.show()

if __name__ == '__main__':
    main()
