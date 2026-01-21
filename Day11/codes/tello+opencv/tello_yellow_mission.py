import cv2
import time
import numpy as np
import os
from djitellopy import Tello

# --- CONFIGURATION ---
TARGET_HEIGHT_CM = 45
YELLOW_LOWER = np.array([20, 100, 100])  # HSV lower bound for Yellow
YELLOW_UPPER = np.array([35, 255, 255])  # HSV upper bound for Yellow
MIN_YELLOW_AREA = 1000  # Minimum pixel area to confirm it's a ball
CAPTURE_COOLDOWN = 5  # Seconds between photos
OUTPUT_DIR = "yellow_captures"

# Create output folder
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Initialize Drone
print("Connecting to Tello...")
tello = Tello()
try:
    tello.connect()
    print(f"Battery: {tello.get_battery()}%")
    tello.streamon()
except Exception as e:
    print(f"Connection failed: {e}")
    exit()

# --- FLIGHT SEQUENCE ---
print("Taking Off...")
tello.takeoff()

# Tello usually takes off to ~100cm. We move down to reach ~45cm.
# Note: Flying low (45cm) creates 'ground effect' turbulence.
print(f"Adjusting height to {TARGET_HEIGHT_CM} cm...")
tello.move_down(55)
time.sleep(2)  # Stabilize

print("Searching for Yellow Ball...")
frame_read = tello.get_frame_read()
last_capture_time = 0

try:
    while True:
        # 1. Get Video Frame
        frame = frame_read.frame
        # Resize for speed
        frame = cv2.resize(frame, (640, 480))

        # 2. Image Processing (Find Yellow)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, YELLOW_LOWER, YELLOW_UPPER)

        # Clean noise (Optional)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        # 3. Find Contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detected = False
        if contours:
            # Find the largest yellow object
            c = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(c)

            if area > MIN_YELLOW_AREA:
                detected = True
                # Draw bounding box
                x, y, w, h = cv2.boundingRect(c)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 3)
                cv2.putText(frame, "YELLOW BALL DETECTED", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                # 4. Capture Logic
                if time.time() - last_capture_time > CAPTURE_COOLDOWN:
                    timestamp = int(time.time())
                    filename = f"{OUTPUT_DIR}/ball_{timestamp}.jpg"

                    # Save the clean frame (without the green box) or annotated frame
                    cv2.imwrite(filename, frame)
                    print(f"[SUCCESS] Picture saved: {filename}")

                    # Optional: Land immediately after finding it?
                    # tello.land(); break

                    last_capture_time = time.time()

        # Display
        cv2.imshow("Tello Yellow Mission", frame)

        # Manual Landing Control
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Landing...")
            tello.land()
            break
        elif key == ord('l'):  # Emergency Land shortcut
            tello.land()
            break

except KeyboardInterrupt:
    print("Emergency Landing...")
    tello.land()

finally:
    tello.streamoff()
    cv2.destroyAllWindows()