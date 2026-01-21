import cv2
from djitellopy import Tello

# --- Configuration ---
SPEED = 50  # Speed of the drone (0-100)

tello = Tello()
tello.connect()
tello.streamon()

print(f"Battery: {tello.get_battery()}%")


def get_keyboard_input():
    """
    Checks which keys are pressed and sets RC values.
    Returns: [left_right, fwd_back, up_down, yaw]


    """
    lr, fb, ud, yv = 0, 0, 0, 0

    # Check for key presses (using OpenCV's waitKey)
    k = cv2.waitKey(1) & 0xFF

    # Quit
    if k == ord('q'):
        return None

        # Takeoff / Land
    if k == ord('t'): tello.takeoff()
    if k == ord('l'): tello.land()

    # Movement (WASD)
    if k == ord('w'):
        fb = SPEED
    elif k == ord('s'):
        fb = -SPEED

    if k == ord('a'):
        lr = -SPEED
    elif k == ord('d'):
        lr = SPEED

    # Elevation (Arrow Keys often map differently, keeping it simple here)
    # Note: Arrow keys may not detect correctly in standard cv2.waitKey on all OS.
    # Using 'u' for Up and 'j' for Down as reliable backups if arrows fail.
    if k == ord('u'):
        ud = SPEED
    elif k == ord('j'):
        ud = -SPEED

    # Rotation (Yaw) - using Left/Right arrows or Z/X
    if k == ord('z'):
        yv = -SPEED
    elif k == ord('x'):
        yv = SPEED

    return [lr, fb, ud, yv]


try:
    while True:
        # 1. Get Video Frame
        frame = tello.get_frame_read().frame
        frame = cv2.resize(frame, (360, 240))
        cv2.imshow("Tello Command Center", frame)

        # 2. Get Keyboard Commands
        vals = get_keyboard_input()

        if vals is None:  # Quit detected
            break

            # 3. Send RC Control
        # send_rc_control(left_right_velocity, forward_backward_velocity, up_down_velocity, yaw_velocity)
        tello.send_rc_control(vals[0], vals[1], vals[2], vals[3])

except Exception as e:
    print(f"Error: {e}")

finally:
    tello.land()
    tello.streamoff()
    cv2.destroyAllWindows()
