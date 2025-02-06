import cv2
from ultralytics import YOLO
import os

model = YOLO("model/yolo11l.pt")  

input_dir = 'input' # Directory containing the input images
output_dir = 'output'  # Directory to save the cropped images

# Get list of all images in the input directory
image_files = [f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]  # what format will it be in? 

# Process each image in the directory
for image_filename in image_files:
    image_path = os.path.join(input_dir, image_filename)
    print(f'Processing {image_filename}...')  

    # Perform object detection
    results = model(source=image_path, show=True)
    results[0].show()

    # Load the original image
    img = cv2.imread(image_path)

    # Extract bounding boxes
    boxes = results[0].boxes.xyxy.tolist()

    # Iterate through the bounding boxes and save each crop
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        cropped_image = img[int(y1):int(y2), int(x1):int(x2)]
        output_filename = f'croppedImage_{image_filename[:-4]}_{i}.jpg' 
        cv2.imwrite(os.path.join(output_dir, output_filename), cropped_image)
        print(f'Saved {output_filename}')  # Optional: Print out which files are saved

print('All images processed successfully.')
