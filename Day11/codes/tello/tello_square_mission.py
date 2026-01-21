from djitellopy import Tello
import time

tello = Tello()
tello.connect()

print(f"Battery Life: {tello.get_battery()}%")

# Safety check: Don't fly if battery is too low
if tello.get_battery() < 20:
    print("Battery too low for mission. Charge first.")
    exit()

print("Taking off in 3 seconds...")
time.sleep(3)

tello.takeoff()

# Move up to eye level
tello.move_up(50)

# --- The Square Mission ---
# The Tello executes these lines one by one.
# It waits for one to finish before starting the next.

for i in range(4):
    print(f"Leg {i + 1} of 4")
    tello.move_forward(100)  # Move forward 100 cm
    tello.rotate_clockwise(90)  # Turn 90 degrees right

# --- End of Mission ---
print("Mission Complete. Landing...")
tello.land()
