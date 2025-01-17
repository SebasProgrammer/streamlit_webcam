from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("lays_model.pt")

# Export the model to ONNX format
model.export(format="onnx")  # creates 'yolov8n.onnx'
