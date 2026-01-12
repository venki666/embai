import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# 1. Load Data
df = pd.read_csv('imu_log.csv', header=None,
                 names=['timestamp', 'ax', 'ay', 'az', 'gx', 'gy', 'gz', 'roll', 'pitch', 'yaw'])

# 2. Analytic Benchmark
df['roll_analytic'] = np.degrees(np.arctan2(df['ay'], df['az']))

# 3. SVM Regression
X = df[['ax', 'ay', 'az']]
y = df['roll']

# Important: SVMs are sensitive to scale. We MUST use a StandardScaler.
# We use an RBF Kernel (Radial Basis Function) to learn the non-linear curve.
svm_model = make_pipeline(StandardScaler(), SVR(kernel='rbf', C=100, gamma=0.1, epsilon=1.0))
svm_model.fit(X, y)
df['roll_pred_svm'] = svm_model.predict(X)

# 4. Evaluation
r2_analytic = r2_score(df['roll'], df['roll_analytic'])
r2_svm = r2_score(df['roll'], df['roll_pred_svm'])

print("--- SVM RESULTS ---")
print(f"R2 Score (Analytic Formula): {r2_analytic:.4f}")
print(f"R2 Score (SVM Regression):   {r2_svm:.4f}")
print("Model Parameters:", svm_model.steps[1][1].get_params())

# 5. Visual 1: Time Series
plt.figure(figsize=(12, 5))
plt.plot(df['timestamp'], df['roll'], 'k-', label='Ground Truth', alpha=0.5)
plt.plot(df['timestamp'], df['roll_analytic'], 'b:', label='Analytic', alpha=0.8)
plt.plot(df['timestamp'], df['roll_pred_svm'], 'r--', label='SVM Prediction', alpha=0.8)
plt.title('SVM vs Analytic Prediction')
plt.xlabel('Timestamp')
plt.ylabel('Roll (Degrees)')
plt.legend()
plt.grid(True)
plt.savefig('svm_timeseries.png')
plt.show()

# 6. Visual 2: The "Decision Surface" (Heatmap)
# We visualize how the SVM "thinks" by plotting its predictions over the Ay/Az plane.
ay_range = np.linspace(df['ay'].min(), df['ay'].max(), 100)
az_range = np.linspace(df['az'].min(), df['az'].max(), 100)
ay_grid, az_grid = np.meshgrid(ay_range, az_range)
# Assume Ax is average
ax_fixed = np.full_like(ay_grid, df['ax'].mean())

X_grid = pd.DataFrame(np.column_stack([ax_fixed.ravel(), ay_grid.ravel(), az_grid.ravel()]),
                      columns=['ax', 'ay', 'az'])
roll_grid = svm_model.predict(X_grid).reshape(ay_grid.shape)

plt.figure(figsize=(8, 6))
contour = plt.contourf(ay_grid, az_grid, roll_grid, levels=50, cmap='viridis')
plt.colorbar(contour, label='Predicted Roll')
plt.scatter(df['ay'], df['az'], c=df['roll'], cmap='viridis', edgecolors='k', s=30, label='Actual Data')
plt.title('SVM Regression Manifold (Learned Physics)')
plt.xlabel('Ay')
plt.ylabel('Az')
plt.savefig('svm_manifold.png')
plt.show()