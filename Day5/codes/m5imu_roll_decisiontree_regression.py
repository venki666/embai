import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# 1. Load Data
df = pd.read_csv('imu_log.csv', header=None,
                 names=['timestamp', 'ax', 'ay', 'az', 'gx', 'gy', 'gz', 'roll', 'pitch', 'yaw'])

# 2. Analytic Calculation (Physics Benchmark)
df['roll_analytic'] = np.degrees(np.arctan2(df['ay'], df['az']))

# 3. Decision Tree Regression
X = df[['ax', 'ay', 'az']]
y = df['roll']

# We limit depth to 3 to keep the visualization readable.
# Deeper trees fit better but are harder to visualize.
dt_model = DecisionTreeRegressor(max_depth=3, random_state=42)
dt_model.fit(X, y)
df['roll_pred_tree'] = dt_model.predict(X)

# 4. Evaluation
r2_analytic = r2_score(df['roll'], df['roll_analytic'])
r2_tree = r2_score(df['roll'], df['roll_pred_tree'])

print("--- RESULTS ---")
print(f"R2 Score (Analytic Formula): {r2_analytic:.4f}")
print(f"R2 Score (Decision Tree):    {r2_tree:.4f}")

# 5. Visual 1: Time Series Comparison
plt.figure(figsize=(12, 5))
plt.plot(df['timestamp'], df['roll'], 'k-', label='Ground Truth (Fusion)', alpha=0.6)
plt.plot(df['timestamp'], df['roll_analytic'], 'b:', label='Analytic Formula')
plt.plot(df['timestamp'], df['roll_pred_tree'], 'r--', label='Decision Tree')
plt.title('Roll Prediction: Decision Tree vs Physics')
plt.legend()
plt.grid(True)
plt.savefig('dt_timeseries.png')
plt.show()

# 6. Visual 2: The Decision Tree Structure
plt.figure(figsize=(20, 10))
plot_tree(dt_model, feature_names=['ax', 'ay', 'az'], filled=True, rounded=True, fontsize=10)
plt.title("Regression Tree Logic (Depth=3)")
plt.savefig('dt_structure.png')
plt.show()