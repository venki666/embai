import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import librosa
import librosa.display
import wave
import os

# Load audio file
def load_audio(file_path):
    audio, sr = librosa.load(file_path, sr=None)
    return audio, sr

# Apply a low-pass, high-pass, or band-pass filter
def apply_filter(audio, sr, filter_type='lowpass', cutoff_freq=1000, order=4):
    nyquist = 0.5 * sr
    normal_cutoff = cutoff_freq / nyquist

    if filter_type == 'lowpass':
        b, a = signal.butter(order, normal_cutoff, btype='low')
    elif filter_type == 'highpass':
        b, a = signal.butter(order, normal_cutoff, btype='high')
    else:
        raise ValueError("Invalid filter type! Choose 'lowpass', 'highpass', or 'bandpass'.")
    
    filtered_audio = signal.filtfilt(b, a, audio)
    return filtered_audio


def apply_bandpass_filter(audio, sr, low_freq, high_freq, order=4):
    nyquist = 0.5 * sr
    low = low_freq / nyquist
    high = high_freq / nyquist
    b, a = signal.butter(order, [low, high], btype='band')
    filtered_audio = signal.filtfilt(b, a, audio)
    return filtered_audio



# Generate FFT and plot the frequency domain
def plot_fft(audio, sr):
    # Compute FFT
    N = len(audio)
    fft = np.fft.fft(audio)
    freqs = np.fft.fftfreq(N, 1/sr)
    magnitude = np.abs(fft)[:N//2]
    freqs = freqs[:N//2]
    
    plt.figure(figsize=(10, 6))
    plt.plot(freqs, magnitude)
    plt.title('FFT of Audio Signal')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.grid(True)
    #plt.show()

# Generate Spectrogram using librosa
def plot_spectrogram(audio, sr):
    plt.figure(figsize=(10, 6))
    D = librosa.amplitude_to_db(librosa.stft(audio), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.title('Spectrogram')
    plt.colorbar(format='%+2.0f dB')
    plt.show()

# Audio Pattern Detection (Silence, Clapping, Snapping, Knocking)
def detect_patterns(audio, sr):
    # Simple silence detection based on thresholding
    silence_threshold = 0.01  # Set a low threshold for silence detection
    silence = np.where(np.abs(audio) < silence_threshold, 1, 0)
    
    # Detect events (e.g., clapping, snapping, knocking) based on peaks in signal
    peaks, _ = signal.find_peaks(np.abs(audio), height=0.2)  # Height for detecting significant peaks
    
    # Visualization of detected peaks (could be clapping, snapping, or knocking)
    plt.figure(figsize=(10, 6))
    plt.plot(audio)
    plt.plot(peaks, audio[peaks], 'ro', label="Detected Peaks")
    plt.title('Pattern Detection (Claps, Snaps, or Knocks)')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.legend()
    #plt.show()

    # Basic analysis: print detected patterns
    print(f"Silence detected at indices: {np.where(silence == 1)[0]}")
    print(f"Detected peaks at indices: {peaks}")

# Main function to execute the pipeline
def process_audio(file_path):
    audio, sr = load_audio(file_path)

    # Apply filters (example: lowpass filter with cutoff at 1000 Hz)
    #filtered_audio = apply_filter(audio, sr, filter_type='bandpass', cutoff_freq=1000)
    filtered_audio = apply_bandpass_filter(audio, sr, 800, 3000)
    
    # Generate FFT and Spectrogram
    plot_fft(filtered_audio, sr)
    plot_spectrogram(filtered_audio, sr)
    plt.show()
    
    # Detect audio patterns
    #detect_patterns(filtered_audio, sr)

# Run the script
if __name__ == "__main__":
    # Specify your audio file path
    file_path = 'original.wav'  # Replace with your actual file path
    if os.path.exists(file_path):
        process_audio(file_path)
    else:
        print(f"File {file_path} not found. Please provide a valid path.")
