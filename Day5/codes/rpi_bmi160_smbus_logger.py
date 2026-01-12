import time
import math
import csv
import threading
import sys
from BMI160_i2c import Driver
from smbus2 import SMBus


# Installs
# sudo apt-get update
# sudo apt-get install python3-pip i2c-tools
# pip3 install smbus2 BMI160-i2c
# --- CONFIGURATION ---
I2C_BUS_ID = 1  # Raspberry Pi usually uses bus 1
I2C_ADDRESS = 0x68  # Default BMI160 address (try 0x69 if this fails)
SAMPLE_RATE_HZ = 100  # 100 Hz Loop
DT = 1.0 / SAMPLE_RATE_HZ
ALPHA = 0.2  # Smoothing Factor
FUSION_ALPHA = 0.96  # Complementary Filter Factor

# Global Flags
is_running = True
is_recording = False
log_filename = "bmi160_log.csv"


# --- HELPER FUNCTIONS ---

def calculate_roll_pitch(ax, ay, az):
    # Avoid division by zero
    hypot = math.sqrt(ay * ay + az * az)
    if hypot == 0: hypot = 0.0001

    # Roll: Rotation around X-axis
    roll_acc = math.degrees(math.atan2(ay, az))

    # Pitch: Rotation around Y-axis
    pitch_acc = math.degrees(math.atan2(-ax, hypot))

    return roll_acc, pitch_acc


def input_thread():
    """ Runs in background to check for 's' key """
    global is_recording, is_running
    print("\n--- CONTROLS ---")
    print("Press 's' + ENTER to Toggle Recording")
    print("Press 'q' + ENTER to Quit")
    print("----------------")

    while is_running:
        try:
            cmd = input()
            if cmd.lower() == 's':
                is_recording = not is_recording
                if is_recording:
                    print(f"\n[RECORDING STARTED] >>> {log_filename}")
                else:
                    print("\n[RECORDING STOPPED]")
            elif cmd.lower() == 'q':
                is_running = False
                print("\nQuitting...")
        except EOFError:
            break


# --- MAIN EXECUTION ---

def main():
    global is_running

    print("Initializing BMI160 via smbus2...")

    try:
        # Initialize Driver (automatically uses smbus2 under the hood)
        sensor = Driver(addr=I2C_ADDRESS, bus=I2C_BUS_ID)
        print("Sensor Connected!")
    except Exception as e:
        print(f"Error: Could not connect to BMI160. Check wiring or I2C address.\n{e}")
        return

    # Create/Overwrite CSV Header
    with open(log_filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "ax_s", "ay_s", "az_s", "gx_s", "gy_s", "gz_s", "roll", "pitch", "yaw"])

    # Start Input Thread
    t = threading.Thread(target=input_thread)
    t.daemon = True
    t.start()

    # Filter Variables
    ax_s, ay_s, az_s = 0.0, 0.0, 0.0
    gx_s, gy_s, gz_s = 0.0, 0.0, 0.0
    roll, pitch, yaw = 0.0, 0.0, 0.0

    print("Sensor Loop Running... (Press 's' to Record)")

    start_time = time.time()
    next_loop = start_time

    while is_running:
        try:
            # --- READ DATA ---
            # The library returns a list [gx, gy, gz, ax, ay, az]
            # Units: Gyro in deg/s?? check library output. usually raw integer or converted.
            # BMI160-i2c lib usually returns raw integers (-32768 to 32767).
            # We must convert them based on default ranges (Accel: 2G, Gyro: 250dps).

            data = sensor.getMotion6()

            # RAW VALUES (Integers)
            gx_raw_int = data[0]
            gy_raw_int = data[1]
            gz_raw_int = data[2]
            ax_raw_int = data[3]
            ay_raw_int = data[4]
            az_raw_int = data[5]

            # CONVERSION (Default Config of BMI160-i2c library)
            # Accel Range: +/- 2G  => 16384 LSB/g
            # Gyro Range:  +/- 250 dps => 131 LSB/dps

            ax_raw = ax_raw_int / 16384.0
            ay_raw = ay_raw_int / 16384.0
            az_raw = az_raw_int / 16384.0

            gx_raw = gx_raw_int / 131.0
            gy_raw = gy_raw_int / 131.0
            gz_raw = gz_raw_int / 131.0

            # --- SMOOTHING ---
            ax_s = (ALPHA * ax_raw) + ((1.0 - ALPHA) * ax_s)
            ay_s = (ALPHA * ay_raw) + ((1.0 - ALPHA) * ay_s)
            az_s = (ALPHA * az_raw) + ((1.0 - ALPHA) * az_s)

            gx_s = (ALPHA * gx_raw) + ((1.0 - ALPHA) * gx_s)
            gy_s = (ALPHA * gy_raw) + ((1.0 - ALPHA) * gy_s)
            gz_s = (ALPHA * gz_raw) + ((1.0 - ALPHA) * gz_s)

            # --- FUSION ---
            roll_acc, pitch_acc = calculate_roll_pitch(ax_s, ay_s, az_s)

            roll = (FUSION_ALPHA * (roll + gx_s * DT)) + ((1.0 - FUSION_ALPHA) * roll_acc)
            pitch = (FUSION_ALPHA * (pitch + gy_s * DT)) + ((1.0 - FUSION_ALPHA) * pitch_acc)
            yaw = yaw + (gz_s * DT)  # Drifts over time

            # --- LOGGING ---
            current_ts = time.time() - start_time

            # Console Output (Throttle to ~5Hz for readability)
            if int(current_ts * 100) % 20 == 0:
                status = "[REC]" if is_recording else "[IDLE]"
                sys.stdout.write(f"\r{status} R:{roll:6.1f} P:{pitch:6.1f} Y:{yaw:6.1f}  ")
                sys.stdout.flush()

            if is_recording:
                with open(log_filename, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        f"{current_ts:.3f}",
                        f"{ax_s:.3f}", f"{ay_s:.3f}", f"{az_s:.3f}",
                        f"{gx_s:.3f}", f"{gy_s:.3f}", f"{gz_s:.3f}",
                        f"{roll:.2f}", f"{pitch:.2f}", f"{yaw:.2f}"
                    ])

        except Exception as e:
            # Catch I2C read errors (happens occasionally)
            pass

        # Maintain 100Hz Loop
        next_loop += DT
        sleep_time = next_loop - time.time()
        if sleep_time > 0:
            time.sleep(sleep_time)


if __name__ == "__main__":
    main()