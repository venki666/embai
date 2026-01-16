import sounddevice as sd
import numpy as np
import librosa
import queue
import sys

# --- Configuration ---
SAMPLE_RATE = 22050  # Must match training data (Librosa default)
DURATION = 0.5  # Seconds (same as training)
N_MFCC = 40  # Same as training
CHANNELS = 1  # Mono audio

# Audio buffer queue
q = queue.Queue()


def audio_callback(indata, frames, time, status):
    """Callback function for the audio stream."""
    if status:
        print(status, file=sys.stderr)
    q.put(indata.copy())


def get_live_features(model_type='dnn'):
    """
    Captures audio and returns extracted features.
    model_type: 'dnn' (returns 1D mean array) or 'cnn' (returns 2D array)
    """
    # Calculate frames per block
    block_size = int(SAMPLE_RATE * DURATION)

    print(f"Recording {DURATION}s segments... Press Ctrl+C to stop.")

    # Start InputStream
    with sd.InputStream(samplerate=SAMPLE_RATE, blocksize=block_size,
                        channels=CHANNELS, callback=audio_callback):
        while True:
            # Get raw audio data from queue
            audio_data = q.get()

            # Flatten to 1D array (librosa expects shape (n,))
            audio_flat = audio_data.flatten()

            # Extract MFCCs
            # Shape: (n_mfcc, time_steps)
            mfccs = librosa.feature.mfcc(y=audio_flat, sr=SAMPLE_RATE, n_mfcc=N_MFCC)

            if model_type == 'dnn':
                # DNN expects Mean of MFCCs (1D vector)
                features = np.mean(mfccs.T, axis=0)
                # Reshape for model input (1, 40)
                features = features.reshape(1, -1)

            elif model_type == 'cnn':
                # CNN expects full time sequence (Time_Steps, N_MFCC)
                # Transpose to (time_steps, n_mfcc)
                features = mfccs.T
                # Reshape for model input (1, time_steps, n_mfcc)
                features = features.reshape(1, features.shape[0], features.shape[1])

            yield features


# Test loop if running standalone
if __name__ == "__main__":
    try:
        for feat in get_live_features(model_type='dnn'):
            print(f"Feature shape: {feat.shape}")
    except KeyboardInterrupt:
        print("\nStopped.")