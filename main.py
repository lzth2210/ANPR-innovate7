from ultralytics import YOLO
import cv2

model = YOLO("C:\\Users\\Lizeth\\OneDrive\\Escritorio\\v6-ANPR\\runs\\detect\\train\\weights\\best.pt")

cap = cv2.VideoCapture(0)


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    annotated_frame = results[0].plot()
    cv2.imshow("YOLOv8 Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == 27 or cv2.getWindowProperty("YOLOv8 Detection", cv2.WND_PROP_VISIBLE) < 1:  # Press 'esc' to exit
        break

cap.release()
cv2.destroyAllWindows()