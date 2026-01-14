#!/usr/bin/env python3
import sys
import numpy as np
from scipy.io import wavfile
from scipy.signal import hilbert

if len(sys.argv) != 1 + 1:
    print("Usage: python analyze_audio.py <wav_file>")
    sys.exit(1)

wav_path = sys.argv[1]
fs, x = wavfile.read(wav_path)
x = x.astype(np.float32) / 32768.0
duration = len(x) / fs
print(f"Duration: {duration:.2f}s")

# Envelope via Hilbert
analytic = hilbert(x)
env = np.abs(analytic)

# Silence: samples where env < threshold
thr = np.mean(env) + 0.5 * np.std(env)
silent = env < thr * 0.3
silence_ratio = np.sum(silent) / len(env)
print(f"Silence ratio (approx): {silence_ratio:.2%}")

# Simple transient detection: env crossings above high threshold
high_thr = np.mean(env) + 2 * np.std(env)
above = env > high_thr
events = 0
i = 0
while i < len(above):
    if above[i]:
        events += 1
        # skip 0.2s to avoid double-count
        i += int(0.2 * fs)
    else:
        i += 1

print(f"Sharp transients (clap/snap/knock-like): {events}")

