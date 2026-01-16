import os
import numpy as np
import librosa
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
DATASET_PATH = "dataset"  # Root folder containing class subfolders
CLASSES = ['Clap', 'Cough', 'Footsteps', 'Glassbreak', 'Knock']

# Audio Parameters
SEGMENT_DURATION = 0.5  # Seconds
OVERLAP = 0.25  # Seconds
N_MFCC = 40  # Features
SAMPLE_RATE = 22050  # Fixed sample rate


def extract_features_dnn(file_path, label):
    """Segment audio and extract Mean MFCCs for DNN."""
    features = []
    labels = []

    try:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE)

        # Windowing
        n_samples = int(SEGMENT_DURATION * sr)
        step = int((SEGMENT_DURATION - OVERLAP) * sr)

        # If audio is shorter than segment, pad it
        if len(y) < n_samples:
            padding = n_samples - len(y)
            y = np.pad(y, (0, padding), mode='constant')

        for start in range(0, len(y) - n_samples + 1, step):
            segment = y[start:start + n_samples]

            # Extract MFCC
            mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=N_MFCC)

            # DNN Feature: Mean across time (result is 1D array of size 40)
            mfcc_mean = np.mean(mfcc.T, axis=0)

            features.append(mfcc_mean)
            labels.append(label)

    except Exception as e:
        print(f"Error parsing {file_path}: {e}")

    return features, labels


# --- 1. Load Data from Folders ---
print("Loading dataset from folders...")
X_data = []
y_data = []

for class_name in CLASSES:
    folder_path = os.path.join(DATASET_PATH, class_name)
    audio_files = glob.glob(os.path.join(folder_path, "*.wav"))

    print(f"Processing '{class_name}': found {len(audio_files)} files")

    for file_path in audio_files:
        feats, labs = extract_features_dnn(file_path, class_name)
        X_data.extend(feats)
        y_data.extend(labs)

X = np.array(X_data)
y = np.array(y_data)

print(f"\nTotal Segments: {X.shape[0]}")
print(f"Feature Vector Shape: {X.shape[1]}")

# Encode Labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# Split Train/Test
X_train, X_test, y_train, y_test = train_test_split(
    X, y_categorical, test_size=0.2, random_state=42, stratify=y_encoded
)

# --- 2. Build DNN Model ---
model = Sequential([
    Dense(256, activation='relu', input_shape=(N_MFCC,)),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(len(CLASSES), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# --- 3. Train ---
print("\nTraining DNN...")
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# --- 4. Evaluate ---
y_pred = np.argmax(model.predict(X_test), axis=1)
y_true = np.argmax(y_test, axis=1)

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=le.classes_))

# Plot Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('DNN Confusion Matrix')
plt.show()

# --- 5. Export to TFLite ---
print("\nConverting to TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

tflite_filename = "audio_dnn_folder_model.tflite"
with open(tflite_filename, 'wb') as f:
    f.write(tflite_model)

print(f"Success! Model exported to '{tflite_filename}'")