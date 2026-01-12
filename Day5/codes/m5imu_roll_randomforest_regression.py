import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import plot_tree
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# 1. Load Data
df = pd.read_csv('imu_log.csv', header=None,
                 names=['timestamp', 'ax', 'ay', 'az', 'gx', 'gy', 'gz', 'roll', 'pitch', 'yaw'])

# 2. Analytic Calculation (Physics Benchmark)
df['roll_analytic'] = np.degrees(np.arctan2(df['ay'], df['az']))

# 3. Random Forest Regression
X = df[['ax', 'ay', 'az']]
y = df['roll']

# We use 100 trees (n_estimators=100)
# We limit depth to 5 to prevent overfitting and keep the model efficient
rf_model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
rf_model.fit(X, y)
df['roll_pred_rf'] = rf_model.predict(X)

# 4. Evaluation
r2_analytic = r2_score(df['roll'], df['roll_analytic'])
r2_rf = r2_score(df['roll'], df['roll_pred_rf'])

print("--- RESULTS ---")
print(f"R2 Score (Analytic Formula): {r2_analytic:.4f}")
print(f"R2 Score (Random Forest):    {r2_rf:.4f}")

# 5. Visual 1: Time Series Comparison
plt.figure(figsize=(12, 5))
plt.plot(df['timestamp'], df['roll'], 'k-', label='Ground Truth (Fusion)', alpha=0.6)
plt.plot(df['timestamp'], df['roll_analytic'], 'b:', label='Analytic Formula')
plt.plot(df['timestamp'], df['roll_pred_rf'], 'r--', label='Random Forest Prediction')
plt.title('Roll Prediction: Random Forest vs Analytic')
plt.xlabel('Timestamp')
plt.ylabel('Roll (degrees)')
plt.legend()
plt.grid(True)
plt.savefig('rf_timeseries.png')
plt.show()

# 6. Visual 2: Structure of ONE Tree from the Forest
# Since we can't visualize all 100 trees, we show the first one to demonstrate the logic.
plt.figure(figsize=(20, 10))
plot_tree(rf_model.estimators_[0], feature_names=['ax', 'ay', 'az'],
          filled=True, rounded=True, fontsize=10, max_depth=3)
plt.title("Logic of Tree #1 (from Random Forest Ensemble)")
plt.savefig('rf_tree_structure.png')
plt.show()