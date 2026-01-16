import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Generate synthetic dataset
np.random.seed(42)

# FIX 1: Changed 1080 to 1000 to match the size of 'y' (must be same number of samples)
# FIX 2: Removed space after 'rand'
X = np.random.rand(1000, 10).astype(np.float32)  # 1000 samples, 10 features

# FIX 3: Removed the stray '|' character at the end of the line
y = np.random.randint(0, 2, size=(1000,)).astype(np.float32)  # Binary labels (0 or 1)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the multi-layer ANN model using Keras
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Display the model summary
# FIX 4: Removed space between 'model.' and 'summary'
model.summary()

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
y_pred = model.predict(X_test)
y_pred_binary = np.round(y_pred).flatten()
accuracy = accuracy_score(y_test, y_pred_binary)
print(f"Accuracy on test set: {accuracy}")