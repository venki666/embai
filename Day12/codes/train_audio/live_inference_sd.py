#!/usr/bin/env python3
"""
Clean, robust mic inference loop using sounddevice that avoids
"Exception ignored from cffi callback ... finished_callback_wrapper"
by:
  - using a context-managed InputStream (guaranteed close)
  - keeping stream alive with a try/finally
  - never throwing exceptions inside the audio callback
  - doing ML inference in the main thread (not inside callback)

Prereqs:
  pip install sounddevice numpy librosa tensorflow

Files expected:
  models/audio_cmd_mfcc.keras
  models/label_map.json
"""

import json
import queue
from collections import deque
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import sounddevice as sd
import librosa
import tensorflow as tf


# -----------------------------
# Paths
# -----------------------------
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "audio_cmd_mfcc.keras"
META_PATH = MODEL_DIR / "label_map.json"

# -----------------------------
# Runtime / Inference config
# -----------------------------
DEVICE: Optional[int] = 1  # set to an int index if needed; else None
CHANNELS = 1
DTYPE = "float32"

MIN_CONFIDENCE = 0.70
SMOOTHING_N = 3  # majority vote window
PRINT_TOPK = 3

# Streaming / buffering
# We'll collect exactly 1.0 sec worth of audio, then run inference.
QUEUE_MAX = 30  # prevents unbounded growth if inference is slow


def majority_vote(labels):
    if not labels:
        return None
    vals, counts = np.unique(np.asarray(labels), return_counts=True)
    return vals[int(np.argmax(counts))]


def load_model_and_meta() -> Tuple[tf.keras.Model, dict]:
    if not MODEL_PATH.exists() or not META_PATH.exists():
        raise FileNotFoundError("Missing model files. Train first to create models/audio_cmd_mfcc.keras and models/label_map.json")
    with open(META_PATH, "r") as f:
        meta = json.load(f)
    model = tf.keras.models.load_model(MODEL_PATH)
    return model, meta


def compute_mfcc(y: np.ndarray, meta: dict) -> np.ndarray:
    sr = int(meta["sr"])
    n_mfcc = int(meta["n_mfcc"])
    n_fft = int(meta["n_fft"])
    hop_length = int(meta["hop_length"])
    win_length = int(meta["win_length"])
    frames = int(meta["frames"])

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
    ).astype(np.float32).T  # (frames_dyn, n_mfcc)

    # pad/trim to training frames
    if mfcc.shape[0] < frames:
        pad = np.zeros((frames - mfcc.shape[0], mfcc.shape[1]), dtype=np.float32)
        mfcc = np.vstack([mfcc, pad])
    elif mfcc.shape[0] > frames:
        mfcc = mfcc[:frames, :]

    return mfcc


def preprocess_for_model(y: np.ndarray, meta: dict) -> np.ndarray:
    # Normalize exactly as training
    mean = np.array(meta["mfcc_mean"], dtype=np.float32).reshape(1, -1)
    std = np.array(meta["mfcc_std"], dtype=np.float32).reshape(1, -1)

    mfcc = compute_mfcc(y, meta)
    mfcc = (mfcc - mean) / (std + 1e-8)

    # (1, frames, n_mfcc, 1)
    x = mfcc[np.newaxis, ..., np.newaxis]
    return x


def safe_audio_callback(indata, frames, time, status, q: queue.Queue):
    """
    MUST NEVER raise exceptions.
    Keep callback light: just copy audio into a queue.
    """
    try:
        if status:
            # Non-fatal stream status; don't raise.
            # You can print(status) if debugging, but keep quiet by default.
            pass

        # indata shape: (frames, channels)
        x = np.asarray(indata, dtype=np.float32)
        if x.ndim == 2:
            x = x[:, 0]  # mono
        else:
            x = x.reshape(-1)

        try:
            q.put_nowait(x.copy())
        except queue.Full:
            # Drop audio if inference is slower than realtime
            pass

    except Exception:
        # Absolutely never let callback throw.
        return


def main():
    model, meta = load_model_and_meta()

    sr = int(meta["sr"])
    num_samples = int(meta["num_samples"])
    labels = list(meta["labels"])

    print("Loaded model.")
    print(f"Sample rate: {sr} Hz | Window: {num_samples/sr:.2f}s | Labels: {labels}")
    print("Press Ctrl+C to stop.\n")

    # Queue holding small audio chunks from callback
    q: queue.Queue = queue.Queue(maxsize=QUEUE_MAX)

    # Rolling buffer to accumulate exactly 1 sec
    buf = np.zeros((0,), dtype=np.float32)

    # For smoothing predictions
    recent = deque(maxlen=SMOOTHING_N)

    try:
        with sd.InputStream(
            samplerate=sr,
            channels=CHANNELS,
            dtype=DTYPE,
            callback=lambda indata, frames, time, status: safe_audio_callback(indata, frames, time, status, q),
            device=DEVICE,
            blocksize=0,  # let PortAudio choose
        ):
            while True:
                chunk = q.get()  # blocking wait
                if chunk.size == 0:
                    continue

                buf = np.concatenate([buf, chunk], axis=0)

                # Process as many full 1s windows as we have
                while buf.size >= num_samples:
                    y = buf[:num_samples]
                    buf = buf[num_samples:]  # hop = 1.0s; change to overlap if desired

                    # Light sanity clipping
                    y = np.clip(y, -1.0, 1.0)

                    # Feature + inference (main thread)
                    x = preprocess_for_model(y, meta)
                    probs = model.predict(x, verbose=0)[0]

                    top_ids = np.argsort(probs)[::-1][:PRINT_TOPK]
                    best_id = int(top_ids[0])
                    best_label = labels[best_id]
                    best_conf = float(probs[best_id])

                    if best_conf < MIN_CONFIDENCE:
                        out_label = "unknown"
                    else:
                        out_label = best_label

                    recent.append(out_label)
                    stable = majority_vote(list(recent)) or out_label

                    topk_str = "  ".join([f"{labels[i]}:{probs[i]:.2f}" for i in top_ids])
                    print(f"pred={out_label:8s}  conf={best_conf:.2f}  stable={stable:8s}  topk=({topk_str})")

    except KeyboardInterrupt:
        print("\nStopping... (Ctrl+C)")
    finally:
        # Ensures PortAudio stream is stopped/closed before Python teardown.
        try:
            sd.stop()
        except Exception:
            pass


if __name__ == "__main__":
    main()
