from ultralytics import YOLO
import cv2 # type: ignore
# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection
    results = model(frame)
    annotated_frame = results[0].plot()

    # Show video with detected object names
    cv2.imshow("YOLOv8 Object Detection", annotated_frame)
    # Press ESC to quit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()