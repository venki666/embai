import time
import numpy as np
import sounddevice as sd
import tflite_runtime.interpreter as tflite

# --- CONFIGURATION ---
MODEL_PATH = "model.tflite"
SAMPLE_RATE = 8000  # Must match your training script (8kHz)
DURATION = 1.0  # Duration in seconds
THRESHOLD = 0.5  # Confidence threshold
# Define your labels exactly as they were in your training dataset
LABELS = ["down", "land", "left", "right", "on", "up", "off", "_background_noise_"]


def main():
    # 1. Load TFLite Model
    print(f"[INFO] Loading {MODEL_PATH}...")
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    # Get input/output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_index = input_details[0]['index']
    output_index = output_details[0]['index']

    # Check expected input type (Int8 vs Float32)
    input_type = input_details[0]['dtype']
    print(f"[INFO] Model expects input type: {input_type}")

    # 2. Setup Audio Stream
    # We calculate the number of samples required (e.g., 8000)
    expected_samples = int(SAMPLE_RATE * DURATION)

    print(f"[INFO] Listening on USB Mic ({SAMPLE_RATE}Hz)...")
    print("      Press Ctrl+C to stop.")

    try:
        while True:
            # --- Capture Audio ---
            # sounddevice.rec records into a NumPy array
            # We use blocking=True to simplify the loop
            audio_buffer = sd.rec(
                frames=expected_samples,
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype='float32',  # Capture as float first
                blocking=True
            )

            # --- Preprocess ---
            # Remove extra dimensions (samples, 1) -> (samples,)
            input_data = audio_buffer.flatten()

            # CONVERSION: Float32 (-1.0 to 1.0) -> Int8 (-128 to 127)
            # Your model was trained with Int8 quantization for M5Core2
            if input_type == np.int8:
                # Scale float to int8 range
                input_data = (input_data * 127).astype(np.int8)
            elif input_type == np.float32:
                # If you trained a non-quantized model, keep as float32
                input_data = input_data.astype(np.float32)

            # Reshape to match model input (1, 8000) or (1, 8000, 1)
            # We use the shape from input_details to be safe
            model_input = np.reshape(input_data, input_details[0]['shape'])

            # --- Inference ---
            interpreter.set_tensor(input_index, model_input)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_index)

            # --- Result ---
            # output_data is usually [[prob1, prob2, ...]]
            # If model is int8, output is also int8 (-128 to 127).
            # We must convert it back to probability (0.0 to 1.0) approx.

            if output_details[0]['dtype'] == np.int8:
                # Dequantize for display: (val + 128) / 255 roughly
                # Or just rely on relative magnitude
                prediction = output_data[0]
                max_idx = np.argmax(prediction)
                confidence = (prediction[max_idx] + 128) / 255.0
            else:
                prediction = output_data[0]
                max_idx = np.argmax(prediction)
                confidence = prediction[max_idx]

            # Print if confidence is high
            if confidence > THRESHOLD:
                cmd = LABELS[max_idx]
                print(f"DETECTED: {cmd.upper()} ({confidence:.2f})")

                # TODO: Add your drone control code here
                # if cmd == "takeoff": tello.takeoff()

    except KeyboardInterrupt:
        print("\n[INFO] Stopping...")


if __name__ == "__main__":
    main()