import cv2
import time
import os
from djitellopy import Tello
from ultralytics import YOLO

# --- CONFIGURATION ---
TARGET_HEIGHT_ADDITION = 40  # Tello takes off to ~80cm. We add 40cm to reach ~120cm.
CONFIDENCE_THRESHOLD = 0.6  # Only capture if confident it's a person
OUTPUT_DIR = "mission_captures"

# Ensure output directory exists
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# 1. Initialize YOLO (Standard Nano Model)
# We will look for Class ID 0 ('person')
print("Loading YOLOv8 Model...")
model = YOLO('yolov8n.pt')

# 2. Initialize Tello
print("Connecting to Tello...")
tello = Tello()

try:
    tello.connect()
    print(f"Battery: {tello.get_battery()}%")
    tello.streamon()

    # Allow camera to warm up
    frame_read = tello.get_frame_read()
    time.sleep(2)

    # --- FLIGHT SEQUENCE ---
    print("Taking Off...")
    tello.takeoff()

    # Tello hovers at ~80-90cm by default.
    # We move up to reach approx 120cm.
    print(f"Ascending {TARGET_HEIGHT_ADDITION}cm to reach ~120cm...")
    tello.move_up(TARGET_HEIGHT_ADDITION)

    print("Hovering & Scanning for Person...")

    # --- DETECTION LOOP ---
    while True:
        # Get frame
        frame = frame_read.frame
        # Resize for faster YOLO inference (optional)
        frame = cv2.resize(frame, (640, 480))

        # Run YOLO
        # stream=True is efficient for video
        results = model(frame, stream=True, verbose=False)

        person_detected = False

        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Check Class ID (0 is 'person' in COCO dataset)
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])

                if cls_id == 0 and conf > CONFIDENCE_THRESHOLD:
                    person_detected = True

                    # Draw box for visual feedback
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, "TARGET ACQUIRED", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                    # --- CAPTURE & LAND SEQUENCE ---
                    print(f"Person Detected (Conf: {conf:.2f})!")

                    # Save the photo
                    filename = f"{OUTPUT_DIR}/mission_success_{int(time.time())}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"Photo Saved: {filename}")

                    # Break the loop to trigger landing
                    raise StopIteration

                    # Display video feed
        cv2.imshow("Tello YOLO Mission", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except StopIteration:
    print("Mission Accomplished. Initiating Landing Sequence...")

except KeyboardInterrupt:
    print("Emergency Interrupt!")

except Exception as e:
    print(f"Error: {e}")

finally:
    # Always land safely, even if code crashes
    print("Landing...")
    tello.land()
    tello.streamoff()
    cv2.destroyAllWindows()