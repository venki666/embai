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
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

# --- Configuration ---
# Map your specific filenames to class labels
FILES = {
    'Knock': 'Knock.wav',
    'Clap': 'clap.wav',
    'Footsteps': 'footsteps.wav',
    'Cough': 'cough.wav',
    'Glassbreak': 'glassbreak.wav'
}

SEGMENT_DURATION = 0.5  # Length of each audio chunk in seconds
OVERLAP = 0.25  # Overlap between chunks in seconds
N_MFCC = 40  # Number of MFCC features to extract


def extract_features_from_file(file_path, label, segment_duration, overlap, n_mfcc):
    """
    Loads an audio file, splits it into segments, and extracts MFCC features for each segment.
    """
    features = []
    labels = []

    try:
        # Load audio file (sr=None preserves original sampling rate)
        y, sr = librosa.load(file_path, sr=None)

        # Calculate samples per segment and step size
        n_samples = int(segment_duration * sr)
        step = int((segment_duration - overlap) * sr)

        # Slide a window across the audio file
        for start in range(0, len(y) - n_samples + 1, step):
            end = start + n_samples
            segment = y[start:end]

            # Extract MFCCs (Mel-frequency cepstral coefficients)
            # This turns audio into a heatmap of frequencies over time
            mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=n_mfcc)

            # Take the mean across time to get a single vector per segment for the DNN
            mfcc_mean = np.mean(mfcc.T, axis=0)

            features.append(mfcc_mean)
            labels.append(label)

    except Exception as e:
        print(f"Error processing {file_path}: {e}")

    return features, labels


# --- 1. Data Preparation ---
print("Extracting features...")
X_data = []
y_data = []

for class_name, file_name in FILES.items():
    # Check if file exists to avoid errors
    if os.path.exists(file_name):
        feats, labs = extract_features_from_file(file_name, class_name, SEGMENT_DURATION, OVERLAP, N_MFCC)
        X_data.extend(feats)
        y_data.extend(labs)
        print(f"Processed {class_name}: {len(feats)} samples extracted.")
    else:
        print(f"Warning: File {file_name} not found.")

X = np.array(X_data)
y = np.array(y_data)

# Encode labels (String -> Integer -> One-Hot)
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# Split into Train and Test sets (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_categorical, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"\nTraining Shape: {X_train.shape}")
print(f"Testing Shape: {X_test.shape}")

# --- 2. Build DNN Model ---
model = Sequential([
    # Input layer
    Dense(256, activation='relu', input_shape=(N_MFCC,)),
    Dropout(0.3),  # Dropout helps prevent overfitting

    # Hidden layers
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),

    # Output layer (Softmax for multi-class classification)
    Dense(len(FILES), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# --- 3. Train Model ---
print("\nTraining Model...")
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1
)

# --- 4. Evaluation ---
print("\nEvaluating Model...")
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)

# Metrics
acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

print("\n--------------------------------------")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1 Score:  {f1:.4f}")
print("--------------------------------------")

# --- 5. Visualization ---

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# Loss and Accuracy Curves
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('[Image of Model Accuracy Curve]')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('')
plt.legend()
plt.show()

# --- 6. Export to .h5 for Netron ---
# This is the key line for Netron compatibility
model_filename = 'audio_dnn_model.h5'
model.save(model_filename)
print(f"\nSUCCESS: Model saved to '{model_filename}'")
print("You can now open this file in https://netron.app to visualize the network.")

# --- 7. Export to .h5 for Netron ---
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)