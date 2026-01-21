import cv2
import time
import os
from djitellopy import Tello

# --- Configuration ---
# Create a folder for captured images if it doesn't exist
OUTPUT_DIR = "tello_captures"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Cooldown to prevent spamming photos (in seconds)
CAPTURE_COOLDOWN = 5
last_capture_time = 0

print("Initializing Tello...")
tello = Tello()

# 1. Connect to Drone
try:
    tello.connect()
    print(f"Battery Life: {tello.get_battery()}%")
except Exception as e:
    print(f"Connection Error: {e}")
    exit()

# 2. Start Video Stream
tello.streamon()
frame_read = tello.get_frame_read()

# 3. Load Face Detection Model (Haar Cascade)
# This file is built-in to cv2, we just need to load it
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

print("------------------------------------------------")
print(" VIDEO STREAM STARTED")
print(" Looking for faces...")
print(" Press 'q' to quit.")
print("------------------------------------------------")

try:
    while True:
        # Get the latest frame from Tello
        frame = frame_read.frame

        # Resize for faster processing (Optional, but recommended for Tello)
        # 360p is a good balance between speed and quality
        frame = cv2.resize(frame, (640, 480))

        # Convert to Grayscale (Haar Cascade requires gray images)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect Faces
        # scaleFactor=1.1: Reduces image size by 10% each pass
        # minNeighbors=5: Higher = fewer detections but better quality
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)

        # Draw rectangles and Check Logic
        for (x, y, w, h) in faces:
            # Draw Green Box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # --- CAPTURE LOGIC ---
            current_time = time.time()
            if current_time - last_capture_time > CAPTURE_COOLDOWN:
                # Construct filename
                img_name = f"{OUTPUT_DIR}/face_{int(current_time)}.jpg"

                # Save the ORIGINAL frame (with the box drawn)
                # If you want a clean photo, save 'frame' before cv2.rectangle
                cv2.imwrite(img_name, frame)

                print(f"[CAPTURE] Face detected! Saved: {img_name}")

                # Visual Feedback: Flash the screen text
                cv2.putText(frame, "PHOTO SAVED!", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                last_capture_time = current_time

        # Display the resulting frame
        cv2.imshow("Tello Face Guard", frame)

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    pass

finally:
    # Cleanup
    print("Shutting down...")
    tello.streamoff()
    cv2.destroyAllWindows()
    # tello.land() # Uncomment if you were flying