import cv2
from ultralytics import YOLO


# Load a model
model = YOLO("model/yolo11n.pt")  # build a new model from YAML


results = model(source="archive/image.png", show=True)
results[0].show


# Load the original image
image = "image.png"
img = cv2.imread(image)

# Extract bounding boxes
boxes = results[0].boxes.xyxy.tolist()

# Iterate through the bounding boxes
for i, box in enumerate(boxes):
    x1, y1, x2, y2 = box
    # Crop the object using the bounding box coordinates
    ultralytics_crop_object = img[int(y1):int(y2), int(x1):int(x2)]
    # Save the cropped object as an image
    cv2.imwrite('ultralytics_crop_' + str(i) + '.jpg', ultralytics_crop_object)