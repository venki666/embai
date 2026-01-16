import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.utils import to_categorical

# --- Configuration ---
FILES = {
    'Knock': 'Knock.wav',
    'Clap': 'clap.wav',
    'Footsteps': 'footsteps.wav',
    'Cough': 'cough.wav',
    'Glassbreak': 'glassbreak.wav'
}

SEGMENT_DURATION = 0.5  # Seconds
OVERLAP = 0.25  # Seconds
N_MFCC = 40  # Number of features


def extract_features_cnn(file_path, label, segment_duration, overlap, n_mfcc):
    """
    Extracts features for CNN. Unlike DNN which takes the MEAN,
    CNN takes the full time-series sequence of MFCCs.
    """
    features = []
    labels = []

    try:
        y, sr = librosa.load(file_path, sr=None)
        n_samples = int(segment_duration * sr)
        step = int((segment_duration - overlap) * sr)

        for start in range(0, len(y) - n_samples + 1, step):
            segment = y[start:start + n_samples]

            # Extract MFCCs
            # Shape: (n_mfcc, time_steps)
            mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=n_mfcc)

            # Transpose to (time_steps, n_mfcc) for Keras Conv1D
            mfcc = mfcc.T

            # We need fixed input size for CNN.
            # Let's pad or trim to a fixed length (e.g., based on segment duration)
            # Expected frames approx = (sr * duration) / hop_length (default 512)
            # For 0.5s at 22050Hz, frames ~= 22. Let's fix to a safe number or resize.
            # For simplicity here, we assume segments are similar size or resize:
            # Note: librosa mfcc output width depends on input length.
            # If segment length is constant in samples, mfcc width is constant.

            features.append(mfcc)
            labels.append(label)

    except Exception as e:
        print(f"Error processing {file_path}: {e}")

    return features, labels


# --- 1. Data Preparation ---
print("Extracting features for CNN...")
X_data = []
y_data = []

# First pass to determine max time steps (to pad sequences if needed)
# However, since we cut fixed sample lengths, MFCC width should be constant.
# Let's verify shape on first import.

for class_name, file_name in FILES.items():
    if os.path.exists(file_name):
        feats, labs = extract_features_cnn(file_name, class_name, SEGMENT_DURATION, OVERLAP, N_MFCC)
        X_data.extend(feats)
        y_data.extend(labs)
    else:
        print(f"Warning: File {file_name} not found.")

X = np.array(X_data)
y = np.array(y_data)

print(f"Input Data Shape: {X.shape}")
# Shape should be (N_samples, Time_Steps, N_Features)

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y_categorical, test_size=0.2, random_state=42, stratify=y_encoded
)

# --- 2. Build 1D CNN Model ---
# Input shape: (Time_Steps, N_MFCC)
input_shape = (X.shape[1], X.shape[2])

model = Sequential([
    # Convolutional Layer 1
    Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=input_shape),
    MaxPooling1D(pool_size=2),
    Dropout(0.2),

    # Convolutional Layer 2
    Conv1D(filters=64, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Dropout(0.2),

    # Flatten output to feed into Dense layers
    Flatten(),

    # Dense Layers
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(len(FILES), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# --- 3. Train Model ---
print("\nTraining CNN...")
history = model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# --- 4. Export for Netron ---
model.save("audio_cnn_model.h5")
print("\nSaved model to 'audio_cnn_model.h5'")

# --- 7. Export to .h5 for Netron ---
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

# --- 5. Evaluation ---
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)

# Metrics
print("\n--- CNN Performance Metrics ---")
print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
print(f"Precision: {precision_score(y_true, y_pred, average='weighted', zero_division=0):.4f}")
print(f"Recall:    {recall_score(y_true, y_pred, average='weighted', zero_division=0):.4f}")
print(f"F1 Score:  {f1_score(y_true, y_pred, average='weighted', zero_division=0):.4f}")

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Accuracy/Loss Plot
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('')
plt.legend()
plt.show()