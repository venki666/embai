import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

# 1. Load Data
df = pd.read_csv('imu_log.csv', header=None,
                 names=['timestamp', 'ax', 'ay', 'az', 'gx', 'gy', 'gz', 'roll', 'pitch', 'yaw'])

# 2. Analytic Benchmark
df['roll_analytic'] = np.degrees(np.arctan2(df['ay'], df['az']))

# 3. FeedForward Neural Network (FFNN)
X = df[['ax', 'ay', 'az']]
y = df['roll']

# Standard Scaling is CRITICAL for Neural Networks to converge
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Architecture: Input(3) -> Hidden(64) -> Hidden(32) -> Output(1)
# Solver: Adam (standard optimizer), Activation: ReLU
nn_model = MLPRegressor(hidden_layer_sizes=(64, 32),
                        activation='relu',
                        solver='adam',
                        max_iter=500,
                        random_state=42)

nn_model.fit(X_scaled, y)
df['roll_pred_nn'] = nn_model.predict(X_scaled)

# 4. Evaluation
r2_analytic = r2_score(df['roll'], df['roll_analytic'])
r2_nn = r2_score(df['roll'], df['roll_pred_nn'])

print("--- RESULTS ---")
print(f"R2 Score (Analytic Formula): {r2_analytic:.4f}")
print(f"R2 Score (FeedForward NN):   {r2_nn:.4f}")

# 5. Visual 1: Time Series Comparison
plt.figure(figsize=(12, 5))
plt.plot(df['timestamp'], df['roll'], 'k-', label='Ground Truth', alpha=0.5)
plt.plot(df['timestamp'], df['roll_analytic'], 'b:', label='Analytic Formula', alpha=0.8)
plt.plot(df['timestamp'], df['roll_pred_nn'], 'r--', label='Neural Network', alpha=0.8)
plt.title('Prediction Comparison: FFNN vs Physics')
plt.xlabel('Timestamp')
plt.ylabel('Roll (Degrees)')
plt.legend()
plt.grid(True)
plt.savefig('nn_timeseries.png')
plt.show()

# 6. Visual 2: Learning Curve (Loss over Training Epochs)
plt.figure(figsize=(8, 4))
plt.plot(nn_model.loss_curve_)
plt.title('Model Training Progress (Loss Curve)')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.grid(True)
plt.savefig('nn_loss.png')
plt.show()

# 7. Visual 3: Model Weights Heatmap (Input Layer)
# This shows which input features (Ax, Ay, Az) the neurons are focusing on.
plt.figure(figsize=(10, 3))
plt.imshow(nn_model.coefs_[0], cmap='viridis', aspect='auto')
plt.colorbar(label='Weight Strength')
plt.yticks([0, 1, 2], ['Ax', 'Ay', 'Az'])
plt.xlabel('Hidden Neurons (Layer 1)')
plt.title('Neural Network Weights Visualization (Input -> Hidden Layer)')
plt.tight_layout()
plt.savefig('nn_weights.png')
plt.show()