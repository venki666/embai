#!/usr/bin/env python3
import time
import numpy as np
import pyaudio
from sshkeyboard import listen_keyboard
from scipy.io import wavfile
from scipy.signal import butter, filtfilt

# ---------- CONFIG ----------
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100          # Sample rate
DEV_INDEX = 2         # <-- set this to your I2S device index
DURATION = 5          # seconds to record
# ----------------------------

def butter_filter(data, cutoff, fs=RATE, btype='low', order=4):
    """
    Generic Butterworth filter.
    cutoff: scalar for low/high, list [low, high] for band.
    fs: sample rate.
    btype: 'low', 'high', 'band'.
    """
    nyq = 0.5 * fs
    if np.isscalar(cutoff):
        normal_cutoff = cutoff / nyq
    else:
        normal_cutoff = [c / nyq for c in cutoff]
    b, a = butter(order, normal_cutoff, btype=btype)
    return filtfilt(b, a, data)

# Init audio
p = pyaudio.PyAudio()
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                input_device_index=DEV_INDEX,
                frames_per_buffer=CHUNK)

recording = False

def record_once():
    global recording
    recording = True
    print("Recording 5s...")
    frames = []
    t0 = time.time()
    while time.time() - t0 < DURATION:
        frames.append(stream.read(CHUNK, exception_on_overflow=False))

    # Convert to float32 [-1,1]
    raw_data = np.frombuffer(b"".join(frames), dtype=np.int16).astype(np.float32) / 32768.0

    # Filters (NOTE keyword args so fs is correct)
    lp_data = butter_filter(raw_data, cutoff=1000, fs=RATE, btype='low')         # Low-pass 1 kHz
    bp_data = butter_filter(raw_data, cutoff=[300, 5000], fs=RATE, btype='band') # Band-pass 300–5000 Hz
    hp_data = butter_filter(raw_data, cutoff=80, fs=RATE, btype='high')          # High-pass 80 Hz

    timestamp = int(time.time())
    base = f"rec_key_{timestamp}"

    # Save WAVs in current directory
    wavfile.write(f"{base}_raw.wav", RATE, (raw_data * 32767).astype(np.int16))
    wavfile.write(f"{base}_lp.wav",  RATE, (lp_data * 32767).astype(np.int16))
    wavfile.write(f"{base}_bp.wav",  RATE, (bp_data * 32767).astype(np.int16))
    wavfile.write(f"{base}_hp.wav",  RATE, (hp_data * 32767).astype(np.int16))
    print(f"Saved: {base}_raw.wav, {base}_lp.wav, {base}_bp.wav, {base}_hp.wav")
    recording = False

def on_press(key: str):
    global recording
    # key is a single-character string in sshkeyboard (space is ' ')
    if key == 'r' and not recording:
        record_once()
    elif key.lower() == 'q':
        print("Quitting...")
        from sshkeyboard import stop_listening
        stop_listening()
        stream.stop_stream()
        stream.close()
        p.terminate()
        raise SystemExit

print("R: record 5s, Q: quit...")
listen_keyboard(on_press=on_press)  # no extra kwargs for 2.3.x

