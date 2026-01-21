import cv2
import time
import numpy as np
import os
from djitellopy import Tello
from ultralytics import YOLO

# --- MISSION CONFIGURATION ---
TARGET_HEIGHT_ADDITION = 40  # Tello takes off to ~80cm. Add 40cm to reach 120cm.
CONF_THRESHOLD = 0.5  # YOLO Confidence
OUTPUT_DIR = "yellow_ball_captures"

# HSV Color Definition for "Yellow"
# Adjust these values if your lighting is very bright/dark
YELLOW_LOWER = np.array([20, 100, 100])
YELLOW_UPPER = np.array([35, 255, 255])

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# 1. Initialize YOLO (Nano model)
print("Loading YOLOv8 Model...")
model = YOLO('yolov8n.pt')

# 2. Initialize Tello
print("Connecting to Tello...")
tello = Tello()


# --- HELPER: Is this box yellow? ---
def is_yellow(image_roi):
    """Checks if the cropped image region is primarily yellow."""
    if image_roi.size == 0: return False

    # Convert to HSV
    hsv = cv2.cvtColor(image_roi, cv2.COLOR_BGR2HSV)

    # Create mask
    mask = cv2.inRange(hsv, YELLOW_LOWER, YELLOW_UPPER)

    # Calculate percentage of yellow pixels
    yellow_pixels = cv2.countNonZero(mask)
    total_pixels = image_roi.shape[0] * image_roi.shape[1]

    # If > 30% of the ball is yellow, we accept it
    if total_pixels > 0 and (yellow_pixels / total_pixels) > 0.3:
        return True
    return False


try:
    tello.connect()
    print(f"Battery: {tello.get_battery()}%")
    tello.streamon()

    # Warmup
    frame_read = tello.get_frame_read()
    time.sleep(2)

    # --- FLIGHT SEQUENCE ---
    print("Taking Off...")
    tello.takeoff()

    print(f"Ascending to 120cm (Moving up {TARGET_HEIGHT_ADDITION}cm)...")
    tello.move_up(TARGET_HEIGHT_ADDITION)

    print("Searching for YELLOW BALL...")

    while True:
        frame = frame_read.frame
        # Resize for speed
        frame = cv2.resize(frame, (640, 480))

        # 1. Run YOLO
        # Class 32 is 'sports ball' in COCO dataset
        results = model(frame, stream=True, classes=[32], verbose=False)

        target_found = False

        for result in results:
            boxes = result.boxes
            for box in boxes:
                # 2. Extract Box Coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])

                if conf > CONF_THRESHOLD:
                    # 3. Crop the detected ball (Region of Interest)
                    # Ensure coordinates are within image bounds
                    h, w, _ = frame.shape
                    roi = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]

                    # 4. Check Color Logic
                    if is_yellow(roi):
                        # Calculate Center Location
                        center_x = int((x1 + x2) / 2)
                        center_y = int((y1 + y2) / 2)

                        # --- SUCCESS ACTION ---
                        print(f">>> YELLOW BALL DETECTED at Location: X={center_x}, Y={center_y}")

                        # Draw Visuals
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
                        cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
                        cv2.putText(frame, f"Loc: {center_x},{center_y}", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                        # Capture Photo
                        filename = f"{OUTPUT_DIR}/yellow_ball_{int(time.time())}.jpg"
                        cv2.imwrite(filename, frame)
                        print(f"Photo Saved: {filename}")

                        target_found = True
                        break  # Break out of box loop

            if target_found: break  # Break out of results loop

        # Display Video
        cv2.imshow("Tello Mission", frame)
        cv2.waitKey(1)

        # 5. Land if target was found
        if target_found:
            print("Target acquired. Initiating Landing...")
            break

except Exception as e:
    print(f"Error: {e}")

finally:
    # --- SAFETY LANDING ---
    print("Landing...")
    tello.land()
    tello.streamoff()
    cv2.destroyAllWindows()