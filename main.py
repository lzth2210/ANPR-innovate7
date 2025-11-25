from ultralytics import YOLO
model = YOLO("yolov8n.pt")

data_path = "C:\\Users\\Lizeth\\OneDrive\\Escritorio\\v6-ANPR\\ANPR-Innovate7-2\\data.yaml"
results = model.train(data=data_path, epochs=15, imgsz=512)
