import cv2
from djitellopy import Tello

# 1. Initialize and Connect
tello = Tello()
tello.connect()

print(f"Battery: {tello.get_battery()}%")

# 2. Turn on the video stream
tello.streamon()

# 3. Read and Display Frames
try:
    while True:
        # Get the current frame from the drone
        frame = tello.get_frame_read().frame

        # Resize for faster processing/smaller window (optional)
        frame = cv2.resize(frame, (360, 240))

        # Display the frame using OpenCV
        cv2.imshow("Tello Video Stream", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    pass

# 4. Cleanup
tello.streamoff()
cv2.destroyAllWindows()
# tello.land() # Uncomment if you were flying
