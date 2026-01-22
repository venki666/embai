import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import pathlib

# --- Configuration ---
DATASET_PATH = 'dataset' # Path to your folder
SAMPLE_RATE = 8000       # We downsample to 8kHz to save RAM on ESP32
INPUT_LENGTH = 8000      # 1 second @ 8kHz
BATCH_SIZE = 32
EPOCHS = 20

# 1. Load Dataset
data_dir = pathlib.Path(DATASET_PATH)
commands = np.array(tf.io.gfile.listdir(str(data_dir)))
#commands = commands[commands != 'README.md']
print('Commands:', commands)

train_ds, val_ds = tf.keras.utils.audio_dataset_from_directory(
    directory=data_dir,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    seed=0,
    output_sequence_length=INPUT_LENGTH,
    subset='both')

def squeeze(audio, labels):
    audio = tf.squeeze(audio, axis=-1) # Remove the last dimension
    return audio, labels

train_ds = train_ds.map(squeeze, tf.data.AUTOTUNE)
val_ds = val_ds.map(squeeze, tf.data.AUTOTUNE)

# 2. Build 1D CNN Model
# Input shape: (8000,) -> suitable for raw waveform
model = models.Sequential([
    layers.Input(shape=(INPUT_LENGTH,)),
    layers.Reshape((INPUT_LENGTH, 1)),  # Add channel dim for Conv1D
    layers.Conv1D(8, 32, activation='relu', strides=4), # Downsample quickly
    layers.MaxPooling1D(2),
    layers.Conv1D(16, 16, activation='relu', strides=2),
    layers.MaxPooling1D(2),
    layers.Conv1D(32, 8, activation='relu'),
    layers.GlobalAveragePooling1D(),
    layers.Dense(len(commands), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

# 3. Convert to TFLite (Quantized)
# Crucial for ESP32: Full Integer Quantization
def representative_dataset():
    for audio, label in train_ds.take(100):
        yield [audio]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
# Ensure Input/Output are int8 compatible
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
tflite_model = converter.convert()

# 4. Save as C Header
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

# Use xxd to convert to array (Linux/Mac)
os.system('xxd -i model.tflite > model_data.cc')
print("Model saved to model_data.cc")