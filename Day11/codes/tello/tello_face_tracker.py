import cv2
import time
from djitellopy import Tello
import numpy as np

# --- Configuration ---
w, h = 360, 240  # Image size to resize to (smaller = faster processing)
pid = [0.4, 0.4, 0]  # Proportional gains: [Yaw, Up/Down, Fwd/Back]
pError = 0  # Previous error (for future derivative term if needed)
startCounter = 0  # For delayed flight start

# Range of distance (Area of face) we want to maintain
# If face area < 6200, move forward. If > 6800, move back.
fbRange = [6200, 6800]

# --- Initialize Tello ---
tello = Tello()
tello.connect()
tello.streamon()
print(f"Battery: {tello.get_battery()}%")


# Take off immediately? (Set to True carefully!)
# tello.takeoff()
# tello.move_up(50)

def findFace(img):
    """
    Detects faces and returns the center coordinates and area of the largest face.
   """
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(imgGray, 1.2, 8)

    myFaceListC = []
    myFaceListArea = []

    for (x, y, w_box, h_box) in faces:
        # Draw rectangle around face
        cv2.rectangle(img, (x, y), (x + w_box, y + h_box), (0, 0, 255), 2)

        # Calculate center of the face
        cx = x + w_box // 2
        cy = y + h_box // 2
        area = w_box * h_box

        cv2.circle(img, (cx, cy), 5, (0, 255, 0), cv2.FILLED)

        myFaceListC.append([cx, cy])
        myFaceListArea.append(area)

    if len(myFaceListArea) != 0:
        # Find the largest face (closest one)
        i = myFaceListArea.index(max(myFaceListArea))
        return img, [myFaceListC[i], myFaceListArea[i]]
    else:
        return img, [[0, 0], 0]


def trackFace(info, w, pid, pError):
    """
    Calculates drone velocity commands based on face position.
    """
    area = info[1]
    x, y = info[0]
    fb = 0  # Forward/Back speed

    # 1. PID for Rotation (Yaw)
    # Error is distance from center of X axis
    error = x - w // 2
    # Calculate speed (Proportional * Error)
    speed = pid[0] * error
    # Clamp speed to stay within valid Tello range (-100 to 100)
    speed = int(np.clip(speed, -100, 100))

    # 2. Control for Forward/Back (Distance maintenance)
    if area > fbRange[0] and area < fbRange[1]:
        fb = 0  # In the "sweet spot", don't move
    elif area > fbRange[1]:
        fb = -20  # Too close, back up
    elif area < fbRange[0] and area != 0:
        fb = 20  # Too far, move forward

    # If no face is detected (x=0), stop rotating
    if x == 0:
        speed = 0
        error = 0

        # Send controls: (Left/Right, Fwd/Back, Up/Down, Yaw)
    # currently only using Yaw (speed) and Fwd/Back (fb)
    tello.send_rc_control(0, fb, 0, speed)

    return error


# --- Main Loop ---
try:
    while True:
        # 1. Get Frame
        img = tello.get_frame_read().frame
        img = cv2.resize(img, (w, h))

        # 2. Detect Face
        img, info = findFace(img)

        # 3. Calculate Controls & Move
        # Note: Uncomment tello.takeoff() above to actually fly!
        pError = trackFace(info, w, pid, pError)

        # 4. Display
        cv2.imshow("Surveillance View", img)

        # Press 'q' to land and quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            tello.land()
            break

except KeyboardInterrupt:
    tello.land()

finally:
    tello.streamoff()
    cv2.destroyAllWindows()
