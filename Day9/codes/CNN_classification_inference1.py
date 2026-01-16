import numpy as np
import tflite_runtime.interpreter as tflite
from rpi_record_process import get_live_features

MODEL_PATH = "audio_cnn_model.tflite"  # ← MUST be .tflite, not .h5!
CLASSES = ['Clap', 'Cough', 'Footsteps', 'Glassbreak', 'Knock']

def main():
    print(f"Loading CNN model from {MODEL_PATH}...")
    
    # Load TFLite model
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"Model input shape: {input_details[0]['shape']}")
    print("Ready! Clap/snap/knock near USB mic (2s each)...")
    
    # Continuous loop
    while True:
        try:
            # Record + extract features
            features = get_live_features()
            
            # Run inference
            interpreter.set_tensor(input_details[0]['index'], features)
            interpreter.invoke()
            predictions = interpreter.get_tensor(output_details[0]['index'])[0]
            
            # Results
            predicted_idx = np.argmax(predictions)
            confidence = predictions[predicted_idx]
            print(f"Detected: {CLASSES[predicted_idx]} ({confidence:.2f})")
            
        except KeyboardInterrupt:
            print("\nStopped.")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()

