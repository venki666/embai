import numpy as np
import librosa
import os
import sys

# Try importing standard TensorFlow (PC) first, then fallback to tflite_runtime (Pi/Lightweight)
try:
    import tensorflow.lite as tflite
except ImportError:
    try:
        import tflite_runtime.interpreter as tflite
    except ImportError:
        print("Error: Please install 'tensorflow' or 'tflite-runtime'")
        sys.exit(1)

# --- Configuration ---
# Path to your audio file
TEST_AUDIO_FILE = "test_audio.wav"

# Choose Model Type: 'dnn' or 'cnn'
MODEL_TYPE = 'dnn'
MODEL_PATH = f"audio_{MODEL_TYPE}_model.tflite"

# Audio Params (Must match training exactly)
SAMPLE_RATE = 22050
DURATION = 0.5  # Seconds
OVERLAP = 0.25  # Seconds (50% overlap)
N_MFCC = 40
CLASSES = ['Clap', 'Cough', 'Footsteps', 'Glassbreak', 'Knock']


def preprocess_dnn(segment, sr):
    """
    DNN Preprocessing: Extract MFCCs and take the MEAN to get a 1D vector.
    Target Shape: (1, 40)
    """
    mfccs = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=N_MFCC)
    features = np.mean(mfccs.T, axis=0)
    return features.reshape(1, -1).astype(np.float32)


def preprocess_cnn(segment, sr, target_time_steps):
    """
    CNN Preprocessing: Extract MFCCs, Transpose, and Pad/Crop to match time steps.
    Target Shape: (1, Time_Steps, 40)
    """
    mfccs = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=N_MFCC)
    features = mfccs.T  # Shape: (Time, Features)

    # Pad or Crop time dimension to match model input
    current_steps = features.shape[0]
    if current_steps < target_time_steps:
        pad_width = target_time_steps - current_steps
        features = np.pad(features, ((0, pad_width), (0, 0)), mode='constant')
    else:
        features = features[:target_time_steps, :]

    # Add batch dimension
    return np.expand_dims(features, axis=0).astype(np.float32)


def run_file_inference(audio_path, model_path, model_type):
    if not os.path.exists(audio_path):
        print(f"Error: File '{audio_path}' not found.")
        return

    print(f"--- Processing File: {audio_path} ---")
    print(f"--- Using Model: {model_type.upper()} ({model_path}) ---")

    # 1. Load Model
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_index = input_details[0]['index']
    output_index = output_details[0]['index']

    # Get expected input shape (specifically for CNN time steps)
    expected_shape = input_details[0]['shape']
    target_time_steps = expected_shape[1] if len(expected_shape) == 3 else 0

    # 2. Load Audio
    # sr=None loads original, but we force 22050 to match training
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)

    # Calculate windowing
    n_samples = int(DURATION * sr)
    step = int((DURATION - OVERLAP) * sr)

    total_duration = librosa.get_duration(y=y, sr=sr)
    print(f"Audio Duration: {total_duration:.2f}s")

    # 3. Sliding Window Inference
    predictions = []

    # Iterate through audio
    for i in range(0, len(y) - n_samples + 1, step):
        segment = y[i: i + n_samples]

        # Preprocess based on model type
        if model_type == 'dnn':
            input_data = preprocess_dnn(segment, sr)
        elif model_type == 'cnn':
            input_data = preprocess_cnn(segment, sr, target_time_steps)

        # Inference
        interpreter.set_tensor(input_index, input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_index)

        # Decode result
        pred_idx = np.argmax(output_data[0])
        confidence = output_data[0][pred_idx]
        label = CLASSES[pred_idx]

        # Calculate timestamp for display
        start_time = i / sr
        end_time = (i + n_samples) / sr

        predictions.append(label)

        # Print segment result
        print(f"Time {start_time:.2f}s - {end_time:.2f}s: {label:<10} (Conf: {confidence:.2f})")

    # 4. Final Majority Vote
    if predictions:
        from collections import Counter
        most_common = Counter(predictions).most_common(1)
        print("\n" + "=" * 30)
        print(f"FINAL PREDICTION: {most_common[0][0]}")
        print("=" * 30)
    else:
        print("Audio too short to extract features.")


if __name__ == "__main__":
    # You can change the filename here or pass it as an argument
    filename = "test_audio.wav"

    # Ensure you have the .tflite file