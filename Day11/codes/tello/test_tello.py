from djitellopy import Tello

# Create the Tello object
tello = Tello()

# Connect to the drone
print("Connecting to Tello...")
tello.connect()

# Print the battery level
print(f"Battery Life: {tello.get_battery()}%")

# Optional: Take off and land immediately (Safety Check!)
# Uncomment the next two lines ONLY if you are ready to fly
# tello.takeoff()
# tello.land()
