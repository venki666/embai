import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# 1. Load Data
df = pd.read_csv('imu_log.csv', header=None,
                 names=['timestamp', 'ax', 'ay', 'az', 'gx', 'gy', 'gz', 'roll', 'pitch', 'yaw'])

# 2. Pre-processing
# Convert timestamp to seconds for integration
df['time_sec'] = (df['timestamp'] - df['timestamp'].iloc[0]) / 1000.0
df['dt'] = df['time_sec'].diff().fillna(0)

# --- METHOD A: ACCELEROMETER ONLY ---
# Fast, but noisy
df['roll_accel'] = np.degrees(np.arctan2(df['ay'], df['az']))

# --- METHOD B: GYROSCOPE ONLY (Integration) ---
# Smooth, but drifts away forever
# Roll = Integral(Gyro_X * dt)
initial_roll = df['roll_accel'].iloc[0]
df['roll_gyro'] = initial_roll + (df['gx'] * df['dt']).cumsum()

# --- METHOD C: COMPLEMENTARY FILTER ---
# Mixes 96% Gyro (for smoothness) + 4% Accel (to fix drift)
# Formula: Angle = 0.96 * (Prev_Angle + Gyro*dt) + 0.04 * Accel
alpha = 0.96
roll_comp = [initial_roll]

for i in range(1, len(df)):
    angle_prev = roll_comp[-1]
    gyro_rate = df['gx'].iloc[i]
    dt = df['dt'].iloc[i]
    accel_angle = df['roll_accel'].iloc[i]

    # The Filter
    new_angle = alpha * (angle_prev + gyro_rate * dt) + (1 - alpha) * accel_angle
    roll_comp.append(new_angle)

df['roll_comp'] = roll_comp

# 3. Evaluation
# Note: 'roll' column is the Ground Truth (likely Madgwick Fusion from device)
r2_accel = r2_score(df['roll'], df['roll_accel'])
r2_gyro = r2_score(df['roll'], df['roll_gyro'])
r2_comp = r2_score(df['roll'], df['roll_comp'])

print("--- ROLL ESTIMATION R2 SCORES ---")
print(f"Accel Only (Trigonometry):   {r2_accel:.4f}")
print(f"Gyro Only (Integration):     {r2_gyro:.4f} (Drift causes failure)")
print(f"Complementary Filter:        {r2_comp:.4f}")

# 4. Visualization
plt.figure(figsize=(12, 6))
plt.plot(df['time_sec'], df['roll'], 'k-', label='Ground Truth (Device Fusion)', linewidth=2, alpha=0.5)
plt.plot(df['time_sec'], df['roll_accel'], 'b:', label='Accel Only', alpha=0.5)
plt.plot(df['time_sec'], df['roll_gyro'], 'g--', label='Gyro Only', alpha=0.5)
plt.plot(df['time_sec'], df['roll_comp'], 'r-', label='Complementary Filter (Calculated)')

plt.title('Why Fusion Matters: Combining Gyro & Accel')
plt.xlabel('Time (s)')
plt.ylabel('Roll (degrees)')
plt.ylim(df['roll'].min() - 20, df['roll'].max() + 20)  # Limit view to handle Gyro drift
plt.legend()
plt.grid(True)
plt.show()