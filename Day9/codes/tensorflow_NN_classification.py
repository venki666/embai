import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# --- 1. Data Generation (Same as before) ---
np.random.seed(42)
X = np.random.rand(1000, 10).astype(np.float32)
y = np.random.randint(0, 2, size=(1000,)).astype(np.float32)

# Reshape y to match the model output shape (batch_size, 1)
y = y.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Use tf.data for efficient batching (standard in low-level TF)
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(32)

# --- 2. Define Model Variables (The "Pure" Way) ---
# We manually initialize Weights (W) and Biases (b) for each layer using random values.
# Architecture: 10 -> 64 -> 32 -> 1

# Layer 1: 10 inputs -> 64 neurons
W1 = tf.Variable(tf.random.normal([10, 64], stddev=0.1), name='W1')
b1 = tf.Variable(tf.zeros([64]), name='b1')

# Layer 2: 64 inputs -> 32 neurons
W2 = tf.Variable(tf.random.normal([64, 32], stddev=0.1), name='W2')
b2 = tf.Variable(tf.zeros([32]), name='b2')

# Layer 3 (Output): 32 inputs -> 1 neuron
W3 = tf.Variable(tf.random.normal([32, 1], stddev=0.1), name='W3')
b3 = tf.Variable(tf.zeros([1]), name='b3')


# --- 3. Define Forward Pass ---
def forward_pass(x):
    # Layer 1: Matrix Mul + Bias -> ReLU
    z1 = tf.matmul(x, W1) + b1
    a1 = tf.nn.relu(z1)

    # Layer 2: Matrix Mul + Bias -> ReLU
    z2 = tf.matmul(a1, W2) + b2
    a2 = tf.nn.relu(z2)

    # Layer 3: Matrix Mul + Bias -> Sigmoid
    z3 = tf.matmul(a2, W3) + b3
    output = tf.math.sigmoid(z3)

    return output


# --- 4. Define Loss Function & Optimizer ---
optimizer = tf.optimizers.Adam(learning_rate=0.001)


def compute_loss(y_true, y_pred):
    # Clip values to prevent log(0) errors
    epsilon = 1e-7
    y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
    # Manual Binary Cross Entropy Formula
    loss = -tf.reduce_mean(y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred))
    return loss


# --- 5. Custom Training Loop (Replacing model.fit) ---
epochs = 100

print("Starting training...")
for epoch in range(epochs):
    epoch_loss_avg = tf.keras.metrics.Mean()

    for x_batch, y_batch in train_dataset:
        with tf.GradientTape() as tape:
            # 1. Forward pass
            predictions = forward_pass(x_batch)
            # 2. Calculate loss
            loss = compute_loss(y_batch, predictions)

        # 3. Calculate Gradients (Backpropagation)
        # We ask TF to calculate how 'loss' changes with respect to W1, b1, W2, etc.
        gradients = tape.gradient(loss, [W1, b1, W2, b2, W3, b3])

        # 4. Update Weights (Optimizer Step)
        optimizer.apply_gradients(zip(gradients, [W1, b1, W2, b2, W3, b3]))

        epoch_loss_avg.update_state(loss)

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}: Loss = {epoch_loss_avg.result():.4f}")

# --- 6. Evaluation ---
print("\nEvaluation:")
# Get predictions on test set
test_logits = forward_pass(X_test)
test_preds = np.round(test_logits.numpy()).flatten()

# Ensure y_test is flattened for comparison
y_test_flat = y_test.flatten()

acc = accuracy_score(y_test_flat, test_preds)
print(f"Accuracy on test set: {acc}")