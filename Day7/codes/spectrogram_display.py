import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


def analyze_audio(file_path):
    # 1. Load the Audio File
    # sr=None tells librosa to use the file's native sample rate rather than resampling
    y, sr = librosa.load(file_path, sr=None)

    # 2. Display Audio Characteristics
    duration = librosa.get_duration(y=y, sr=sr)

    print(f"--- Audio Characteristics for {file_path} ---")
    print(f"Sample Rate (SR):   {sr} Hz")
    print(f"Duration:           {duration:.4f} seconds")
    print(f"Total Samples:      {len(y)}")
    print(f"Shape (Channels):   {y.shape} (Mono)" if y.ndim == 1 else f"Shape: {y.shape} (Stereo)")
    print("-" * 40)

    # 3. Generate Representations
    plt.figure(figsize=(14, 8))

    # Top Plot: The Waveform (Time Domain)
    # Shows amplitude changes over time (Good for seeing silence vs. activity)
    plt.subplot(2, 1, 1)
    librosa.display.waveshow(y, sr=sr, alpha=0.6)
    plt.title('Waveform (Time Domain)')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)

    # Bottom Plot: The Spectrogram (Frequency Domain)
    # 1. Compute Short-Time Fourier Transform (STFT)
    D = librosa.stft(y)
    # 2. Convert amplitude to Decibels (Log scale) for visualization
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    plt.subplot(2, 1, 2)
    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz', cmap='magma')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram (Frequency Domain)')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')

    plt.tight_layout()
    plt.show()


# Run the function on your file
file_name = 'rec_23926.wav'
try:
    analyze_audio(file_name)
except FileNotFoundError:
    print(f"Error: The file '{file_name}' was not found. Please check the path.")