from ultralytics import YOLO
import cv2
from collections import defaultdict
import json

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  

# Video file or webcam
video_path = "video.mp4" 
cap = cv2.VideoCapture(video_path)

# Output video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (800, 600))

# Report data
report = {
    "total_frames": 0,
    "total_detections": 0,
    "class_counts": defaultdict(int)
}

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    # Resize frame to 800x600
    frame = cv2.resize(frame, (800, 600))

    # Run YOLO detection
    results = model(frame)

    object_count = 0

    for result in results:
        boxes = result.boxes

        for box in boxes:
            object_count += 1

            # Bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Confidence score
            conf = float(box.conf[0])

            # Class ID and name
            cls = int(box.cls[0])
            class_name = model.names[cls]

            report["total_detections"] += 1
            report["class_counts"][class_name] += 1

            # Draw bounding box
            cv2.rectangle(frame,
                          (x1, y1),
                          (x2, y2),
                          (0, 255, 0),
                          2)

            # Label
            label = f"{class_name} {conf:.2f}"
            cv2.putText(frame,
                        label,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2)

    # Display object count
    cv2.putText(frame,
                f"Objects Count: {object_count}",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                3)
    
    out.write(frame)

    cv2.imshow("YOLOv8 Object Detection & Counting", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

# Convert defaultdict to dict
report["class_counts"] = dict(report["class_counts"])

# Save JSON report
with open("report.json", "w") as f:
    json.dump(report, f, indent=4)

print("Processed video saved as output.mp4")
print("Report saved as report.json")