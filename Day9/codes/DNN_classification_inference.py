import numpy as np
import tensorflow as tf
from rpi_record_process import get_live_features

# --- Configuration ---
MODEL_PATH_DNN = "audio_dnn_model.h5"
#MODEL_PATH_CNN = "audio_cnn_model.h5"

# Define classes in the same order as training (LabelEncoder order)
# usually alphabetical: Clap, Cough, Footsteps, Glassbreak, Knock
CLASSES = ['Clap', 'Cough', 'Footsteps', 'Glassbreak', 'Knock']


def run_inference(model_path, model_type):
    print(f"Loading {model_type.upper()} model from {model_path}...")

    # Load Keras Model
    try:
        interpreter = tf.keras.models.load_model(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print("Model loaded. Listening...")

    # Process audio stream
    try:
        for input_data in get_live_features(model_type=model_type):

            # --- PREDICTION ---
            # Returns array of probabilities, e.g., [[0.1, 0.05, 0.8, ...]]
            prediction_probs = interpreter.predict(input_data, verbose=0)

            # Get index of highest probability
            predicted_index = np.argmax(prediction_probs)
            confidence = prediction_probs[0][predicted_index]

            predicted_label = CLASSES[predicted_index]

            # --- OUTPUT ---
            # Only print if confidence is high enough to reduce noise
            if confidence > 0.6:
                print(f"Detected: {predicted_label} ({confidence * 100:.1f}%)")
            else:
                print(".", end="", flush=True)  # weak signal indicator

    except KeyboardInterrupt:
        print("\nInference stopped.")


if __name__ == "__main__":
    # Choose which model to run: 'dnn' or 'cnn'
    # Ensure you have the corresponding .h5 file in the directory

    # Example: Run DNN
    run_inference(MODEL_PATH_DNN, 'dnn')

    # To run CNN, comment out above and uncomment below:
    # run_inference(MODEL_PATH_CNN, 'cnn')