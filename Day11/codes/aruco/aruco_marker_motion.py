import cv2
import cv2.aruco as aruco
import numpy as np
import time
from djitellopy import Tello

# --- CONFIGURATION ---
TARGET_HEIGHT_ADDITION = 40  # 80cm (Takeoff) + 40cm = ~120cm
MOVE_DIST = 20  # Tello Min distance is 20cm (10cm commands are often ignored)
COOLDOWN = 3  # Seconds to wait between commands

# Define Action Mapping based on your generated IDs
# ID : Action
ACTION_MAP = {
    0: "UP",
    5: "DOWN",
    12: "LEFT",
    42: "RIGHT",
    100: "LAND"
}

# --- SETUP ---
print("Connecting to Tello...")
tello = Tello()
tello.connect()
print(f"Battery: {tello.get_battery()}%")

tello.streamon()
time.sleep(2)  # Warmup

# Setup ArUco Detector (6x6 Dictionary)
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
parameters = aruco.DetectorParameters()

# --- FLIGHT SEQUENCE ---
print("Taking Off...")
tello.takeoff()

print(f"Adjusting altitude to ~120cm (Moving UP {TARGET_HEIGHT_ADDITION}cm)...")
tello.move_up(TARGET_HEIGHT_ADDITION)

print("Scanning for ArUco Markers...")
last_action_time = 0

try:
    frame_read = tello.get_frame_read()

    while True:
        # 1. Read Frame
        frame = frame_read.frame
        # Resize for speed
        frame = cv2.resize(frame, (640, 480))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 2. Detect Markers
        corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        # 3. Process Markers
        if ids is not None and (time.time() - last_action_time > COOLDOWN):
            # Flatten ID list
            detected_ids = ids.flatten()

            # Prioritize the first recognized marker in our map
            for marker_id in detected_ids:
                if marker_id in ACTION_MAP:
                    action = ACTION_MAP[marker_id]
                    print(f">>> Marker {marker_id} Detected! Performing: {action}")

                    # Execute Command
                    if action == "UP":
                        tello.move_up(MOVE_DIST)
                    elif action == "DOWN":
                        tello.move_down(MOVE_DIST)
                    elif action == "LEFT":
                        tello.move_left(MOVE_DIST)
                    elif action == "RIGHT":
                        tello.move_right(MOVE_DIST)
                    elif action == "LAND":
                        print("Landing command received...")
                        tello.land()
                        raise StopIteration  # Exit loop

                    last_action_time = time.time()
                    break  # Execute only one command at a time

        # 4. Draw Markers for Visualization
        if ids is not None:
            aruco.drawDetectedMarkers(frame, corners, ids)

        # Display
        cv2.imshow("Tello ArUco Controller", frame)

        # Manual Landing (Safety)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Manual Landing...")
            tello.land()
            break

except StopIteration:
    print("Mission Complete.")
except Exception as e:
    print(f"Error: {e}")
    tello.land()

finally:
    tello.streamoff()
    cv2.destroyAllWindows()