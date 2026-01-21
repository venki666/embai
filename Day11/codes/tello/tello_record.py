import cv2
import time
from djitellopy import Tello

# --- Settings ---
tello = Tello()
tello.connect()
tello.streamon()

# 1. Setup Video Writer
# We need the frame dimensions to setup the writer correctly
frame_read = tello.get_frame_read()
height, width, _ = frame_read.frame.shape

# Define the codec (mp4v is standard for .mp4) and create VideoWriter object
# Format: 'filename.mp4', codec, fps, (width, height)
video_writer = cv2.VideoWriter('tello_flight.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))

print("Recording... Press 'q' to stop and save.")

try:
    while True:
        # 2. Get the frame
        frame = frame_read.frame

        # 3. Write the frame to the file
        video_writer.write(frame)

        # 4. Show the frame on screen (optional, so you can see what you record)
        cv2.imshow("Recording", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    pass

finally:
    # 5. Save and Cleanup
    print("Saving video file...")
    video_writer.release()  # CRITICAL: This finalizes the file
    tello.streamoff()
    cv2.destroyAllWindows()
