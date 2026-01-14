import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


def analyze_audio_mfcc(file_path):
    # 1. Load the Audio File
    # sr=None ensures we read the native sample rate (important for checking data quality)
    y, sr = librosa.load(file_path, sr=None)

    # 2. Display Audio Characteristics
    duration = librosa.get_duration(y=y, sr=sr)

    print(f"--- Audio Characteristics for {file_path} ---")
    print(f"Sample Rate (SR):   {sr} Hz")
    print(f"Duration:           {duration:.4f} seconds")
    print(f"Total Samples:      {len(y)}")
    print(f"Shape:              {y.shape}")
    print("-" * 40)

    # 3. Generate Representations
    plt.figure(figsize=(12, 8))

    # Plot 1: The Waveform (Time Domain)
    plt.subplot(2, 1, 1)
    librosa.display.waveshow(y, sr=sr, alpha=0.6)
    plt.title(f'Waveform: {file_path}')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)

    # Plot 2: MFCC Spectrogram (Feature Domain)
    # We extract 13 coefficients, which is standard for speech/keyword spotting
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    plt.subplot(2, 1, 2)
    # Visualize the MFCCs
    librosa.display.specshow(mfccs, x_axis='time', sr=sr, cmap='viridis')
    plt.colorbar(format='%+2.0f')
    plt.title('MFCC Spectrogram (13 Coefficients)')
    plt.xlabel('Time (s)')
    plt.ylabel('MFCC Coefficient Index')
    plt.yticks(np.arange(0, 13, 2))  # Show indices 0, 2, 4... for clarity

    plt.tight_layout()
    plt.show()


# Run the function on your attached file
# Ensure the file is in the same directory or provide the full path
file_name = 'rec_23926.wav'

try:
    analyze_audio_mfcc(file_name)
except FileNotFoundError:
    print(f"Error: '{file_name}' not found. Please ensure the file is in your working directory.")