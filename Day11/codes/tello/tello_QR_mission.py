import cv2
import time
from djitellopy import Tello
from pyzbar.pyzbar import decode
import numpy as np

tello = Tello()
tello.connect()
tello.streamon()

print(f"Battery: {tello.get_battery()}%")

# Cooldown to prevent executing the same command 50 times a second
last_command_time = 0
COOLDOWN = 5  # Seconds


def execute_command(cmd):
    global last_command_time

    # Check if enough time has passed since last command
    if time.time() - last_command_time < COOLDOWN:
        return

    print(f"EXECUTING COMMAND: {cmd}")

    if cmd == "takeoff":
        tello.takeoff()
        # Move up slightly to see better
        # tello.move_up(30)

    elif cmd == "land":
        tello.land()

    elif cmd == "flip":
        # Only flip if flying
        try:
            tello.flip_left()
        except:
            print("Cannot flip (maybe not flying?)")

            # Reset timer
    last_command_time = time.time()


try:
    while True:
        frame = tello.get_frame_read().frame
        frame = cv2.resize(frame, (640, 480))

        decoded_objects = decode(frame)

        for obj in decoded_objects:
            qr_data = obj.data.decode("utf-8").lower().strip()

            # Draw visual feedback
            pts = np.array(obj.polygon, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], True, (255, 0, 0), 3)
            cv2.putText(frame, qr_data, (obj.rect.left, obj.rect.top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0),
                        2)

            # Execute the logic
            execute_command(qr_data)

        cv2.imshow("QR Mission Commander", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            tello.land()
            break

except Exception as e:
    print(e)
    tello.land()

finally:
    tello.streamoff()
    cv2.destroyAllWindows()
