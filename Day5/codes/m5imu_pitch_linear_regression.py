import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# 1. Load Data
# timestamp, ax, ay, az, gx, gy, gz, roll, pitch, yaw
df = pd.read_csv('imu_log.csv', header=None,
                 names=['timestamp', 'ax', 'ay', 'az', 'gx', 'gy', 'gz', 'roll', 'pitch', 'yaw'])

# 2. Method A: Analytic Calculation (Physical Formula for Pitch)
# Pitch relates primarily to the X-axis component of gravity
df['pitch_acc_calc'] = np.degrees(np.arctan2(-df['ax'], np.sqrt(df['ay']**2 + df['az']**2)))

# 3. Method B: Linear Regression (Machine Learning)
X = df[['ax', 'ay', 'az']] # Features
y = df['pitch']            # Target

model = LinearRegression()
model.fit(X, y)
df['pitch_pred_linreg'] = model.predict(X)

# 4. Evaluation (R2 Score)
r2_analytic = r2_score(df['pitch'], df['pitch_acc_calc'])
r2_linreg = r2_score(df['pitch'], df['pitch_pred_linreg'])

print("--- Results for Pitch Prediction ---")
print(f"R2 Score (Analytic Formula):   {r2_analytic:.4f}")
print(f"R2 Score (Linear Regression):  {r2_linreg:.4f}")
print(f"Learned Coefficients: {model.coef_}")

# 5. Visualization
plt.figure(figsize=(12, 6))
plt.plot(df['timestamp'], df['pitch'], 'k-', label='Ground Truth (Fusion)', linewidth=1.5, alpha=0.7)
plt.plot(df['timestamp'], df['pitch_pred_linreg'], 'r--', label='Linear Regression Prediction', linewidth=1)
plt.plot(df['timestamp'], df['pitch_acc_calc'], 'b:', label='Analytic Formula', linewidth=1)
plt.title('Pitch Estimation: Machine Learning vs. Trigonometry')
plt.xlabel('Time (ms)')
plt.ylabel('Pitch Angle (Degrees)')
plt.legend()
plt.grid(True)
plt.savefig('pitch_comparison.png')
plt.show()