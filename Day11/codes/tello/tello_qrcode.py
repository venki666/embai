import cv2
from djitellopy import Tello
from pyzbar.pyzbar import decode
import numpy as np

# 1. Initialize Tello
tello = Tello()
tello.connect()
tello.streamon()

print(f"Battery: {tello.get_battery()}%")


def read_qr(img):
    # Decode function looks for QR codes (and barcodes)
    decoded_objects = decode(img)

    for obj in decoded_objects:
        # 1. Extract Data
        # Data comes as bytes, need to decode to string
        qr_data = obj.data.decode("utf-8")
        qr_type = obj.type

        # 2. Draw Box
        # The points of the polygon (usually a square)
        points = obj.polygon

        # If points are detected, draw lines connecting them
        if len(points) == 4:
            pts = np.array(points, dtype=np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(img, [pts], True, (0, 255, 0), 3)

            # 3. Put Text on Screen
        # Locate the text above the QR code
        rect = obj.rect
        cv2.putText(img, qr_data, (rect.left, rect.top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        print(f"Detected: {qr_data}")

    return img


# Main Loop
try:
    while True:
        frame = tello.get_frame_read().frame
        # Resize generally helps pyzbar process faster
        frame = cv2.resize(frame, (640, 480))

        frame = read_qr(frame)

        cv2.imshow("Tello QR Scanner", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    pass

finally:
    tello.streamoff()
    cv2.destroyAllWindows()
