import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np

# --- Configuration ---
FILENAME = "test_audio.wav"  # Name of the output file
DURATION = 5  # Duration in seconds
SAMPLE_RATE = 22050  # Sample rate (Hz). 22050 is standard for ML models.


# Use 44100 for high quality music.

def record_audio():
    print(f"Recording for {DURATION} seconds...")

    # Start recording
    # channels=1 for mono (standard for most ML models), 2 for stereo
    audio_data = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='int16')

    # Wait until recording is finished
    sd.wait()

    print("Recording finished.")

    # Save as WAV file
    write(FILENAME, SAMPLE_RATE, audio_data)
    print(f"Saved to '{FILENAME}'")


if __name__ == "__main__":
    # Check available devices (optional, just to see what's default)
    # print(sd.query_devices())

    record_audio()