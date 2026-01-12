import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# 1. Load Data
df = pd.read_csv('imu_log.csv', header=None,
                 names=['timestamp', 'ax', 'ay', 'az', 'gx', 'gy', 'gz', 'roll', 'pitch', 'yaw'])

# 2. Analytic Formula (Physics-based)
# Roll is the rotation around X-axis, calculated from Y and Z components
df['roll_analytic'] = np.degrees(np.arctan2(df['ay'], df['az']))

# 3. Non-Linear Regression (Polynomial Degree 3)
# We map [ax, ay, az] -> [1, ax, ay, az, ax^2, ax*ay... ax^3...]
X = df[['ax', 'ay', 'az']]
y = df['roll']

poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X)

model = LinearRegression()
model.fit(X_poly, y)
df['roll_pred_poly'] = model.predict(X_poly)

# 4. Evaluation
r2_analytic = r2_score(df['roll'], df['roll_analytic'])
r2_poly = r2_score(df['roll'], df['roll_pred_poly'])

print("--- ROLL RESULTS ---")
print(f"R2 Score (Analytic): {r2_analytic:.4f}")
print(f"R2 Score (Poly Deg 3): {r2_poly:.4f}")
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

# 5. Visualization
plt.figure(figsize=(10, 5))
plt.plot(df['timestamp'], df['roll'], 'k-', label='Ground Truth (Fusion)', alpha=0.6)
plt.plot(df['timestamp'], df['roll_analytic'], 'b:', label='Analytic Formula')
plt.plot(df['timestamp'], df['roll_pred_poly'], 'r--', label='Polynomial Regression')
plt.title('Roll: Non-Linear Regression vs Analytic')
plt.legend()
plt.show()