#!/usr/bin/env python3
import sys
import numpy as np
from scipy.io import wavfile
from scipy.fft import fft, fftfreq
from scipy.signal import spectrogram
import matplotlib.pyplot as plt

if len(sys.argv) != 2:
    print("Usage: python visualize_audio.py <wav_file>")
    sys.exit(1)

wav_path = sys.argv[1]
fs, data = wavfile.read(wav_path)             # int16
data = data.astype(np.float32) / 32768.0      # → [-1, 1]
N = len(data)
t = np.arange(N) / fs

# FFT
yf = fft(data)
xf = fftfreq(N, 1/fs)[:N//2]

plt.figure(figsize=(12, 8))

# 1) Waveform
plt.subplot(3, 1, 1)
plt.plot(t, data)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title(f"Waveform: {wav_path}")
plt.grid(True)

# 2) FFT magnitude
plt.subplot(3, 1, 2)
plt.semilogy(xf[1:], 2.0/N * np.abs(yf[1:N//2]))
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.title("FFT Spectrum")
plt.grid(True)

# 3) Spectrogram
f, tt, Sxx = spectrogram(data, fs, nperseg=1024, noverlap=512)
plt.subplot(3, 1, 3)
plt.pcolormesh(tt, f, 10*np.log10(Sxx + 1e-12), shading="gouraud")
plt.ylabel("Frequency (Hz)")
plt.xlabel("Time (s)")
plt.title("Spectrogram")
plt.colorbar(label="Power (dB)")

plt.tight_layout()
plt.savefig("analysis.png", dpi=150, bbox_inches="tight")
plt.show()

