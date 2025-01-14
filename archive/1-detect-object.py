from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n.pt')  # build a new model from YAML

# Train the model
results = model(source="archive/phone.png", show=True, conf = 0.4, save=True)

