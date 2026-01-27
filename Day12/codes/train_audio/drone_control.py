#!/usr/bin/env python3
"""
Direct voice-command control of RoboMaster Tello using:
- MFCC CNN audio command model (on, off, left, right, up, down [+ optional noise/tello])
- sounddevice mic streaming inference
- DJITelloPy to control the drone

Behavior (ONLY if confidence >= 80%):
- "on"   -> takeoff
- "off"  -> land
- "left" -> move_left(10 cm)
- "right"-> move_right(10 cm)
- "up"   -> move_up(10 cm)
- "down" -> move_down(10 cm)

Notes:
- Tello naturally hovers/holds position between commands.
- This script avoids the cffi callback warning by:
  * context-managed InputStream
  * exception-proof callback
  * ML inference in main thread
- Safety: test in a net / open area; always be ready to land.

Deps:
  pip install numpy librosa sounddevice tensorflow djitellopy

Files expected:
  models/audio_cmd_mfcc.keras
  models/label_map.json
"""

import json
import queue
import time
from pathlib import Path
from typing import Optional, Tuple, Set

import numpy as np
import sounddevice as sd
import librosa
import tensorflow as tf

from djitellopy import Tello


# -----------------------------
# Paths
# -----------------------------
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "audio_cmd_mfcc.keras"
META_PATH = MODEL_DIR / "label_map.json"

# -----------------------------
# Audio / Inference config
# -----------------------------
DEVICE: Optional[int] = None  # set to mic device index if needed
CHANNELS = 1
DTYPE = "float32"

CONF_THRESH = 0.80       # 80% minimum confidence to trigger drone command
MOVE_CM = 10             # default move distance
COMMAND_COOLDOWN_SEC = 1.2   # prevents repeated triggers on same word
PRINT_TOPK = 3

QUEUE_MAX = 40           # audio chunk queue capacity

# Commands we care about
VALID_COMMANDS: Set[str] = {"on", "off", "left", "right", "up", "down"}


# -----------------------------
# Model utilities
# -----------------------------
def load_model_and_meta() -> Tuple[tf.keras.Model, dict]:
    if not MODEL_PATH.exists() or not META_PATH.exists():
        raise FileNotFoundError(
            "Missing model files. Train first to create:\n"
            "  models/audio_cmd_mfcc.keras\n"
            "  models/label_map.json"
        )
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
    mean = np.array(meta["mfcc_mean"], dtype=np.float32).reshape(1, -1)
    std = np.array(meta["mfcc_std"], dtype=np.float32).reshape(1, -1)
    mfcc = compute_mfcc(y, meta)
    mfcc = (mfcc - mean) / (std + 1e-8)
    return mfcc[np.newaxis, ..., np.newaxis]  # (1, frames, n_mfcc, 1)


# -----------------------------
# Audio callback (must be safe)
# -----------------------------
def safe_audio_callback(indata, frames, time_info, status, q: queue.Queue):
    """PortAudio callback: MUST NEVER raise exceptions."""
    try:
        x = np.asarray(indata, dtype=np.float32)
        if x.ndim == 2:
            x = x[:, 0]  # mono
        else:
            x = x.reshape(-1)

        try:
            q.put_nowait(x.copy())
        except queue.Full:
            pass
    except Exception:
        return


def print_prediction(best_label: str, best_conf: float, topk: list[tuple[str, float]]):
    topk_str = "  ".join([f"{l}:{c*100:5.1f}%" for l, c in topk])
    print(f"recognized={best_label:8s}  conf={best_conf*100:5.1f}%  topk=({topk_str})")


# -----------------------------
# Drone control
# -----------------------------
def do_drone_action(tello: Tello, cmd: str):
    """
    All DJITelloPy movement commands are blocking.
    Tello will hover (hold) after the command completes.
    """
    if cmd == "on":
        tello.takeoff()
    elif cmd == "off":
        tello.land()
    elif cmd == "left":
        tello.move_left(MOVE_CM)
    elif cmd == "right":
        tello.move_right(MOVE_CM)
    elif cmd == "up":
        tello.move_up(MOVE_CM)
    elif cmd == "down":
        tello.move_down(MOVE_CM)


def main():
    # ---- Load model/meta ----
    model, meta = load_model_and_meta()
    sr = int(meta["sr"])
    num_samples = int(meta["num_samples"])
    labels = list(meta["labels"])

    # ---- Connect to Tello ----
    tello = Tello()
    print("Connecting to Tello...")
    tello.connect()
    batt = tello.get_battery()
    print(f"Tello connected. Battery: {batt}%")
    try:
        tello.streamoff()
    except Exception:
        pass

    print("\nReady. Say: on/off/left/right/up/down (each spoken as ~1 second).")
    print("Ctrl+C to stop.\n")

    # ---- Audio queue/buffer ----
    q: queue.Queue = queue.Queue(maxsize=QUEUE_MAX)
    buf = np.zeros((0,), dtype=np.float32)

    # ---- Command gating ----
    last_cmd = None
    last_action_time = 0.0

    try:
        with sd.InputStream(
            samplerate=sr,
            channels=CHANNELS,
            dtype=DTYPE,
            callback=lambda indata, frames, time_info, status: safe_audio_callback(indata, frames, time_info, status, q),
            device=DEVICE,
            blocksize=0,
        ):
            while True:
                chunk = q.get()
                if chunk.size == 0:
                    continue
                buf = np.concatenate([buf, chunk], axis=0)

                # Process full 1-second windows (no overlap)
                while buf.size >= num_samples:
                    y = buf[:num_samples]
                    buf = buf[num_samples:]

                    y = np.clip(y, -1.0, 1.0)
                    x = preprocess_for_model(y, meta)

                    probs = model.predict(x, verbose=0)[0]
                    top_ids = np.argsort(probs)[::-1][:PRINT_TOPK]
                    best_id = int(top_ids[0])
                    best_label = labels[best_id]
                    best_conf = float(probs[best_id])
                    topk = [(labels[i], float(probs[i])) for i in top_ids]

                    # Always print what the model thinks (requested)
                    print_prediction(best_label, best_conf, topk)

                    # Only execute drone commands for the 6 target words at >=80%
                    if best_label in VALID_COMMANDS and best_conf >= CONF_THRESH:
                        now = time.time()

                        # Cooldown and "same-command suppression" to prevent repeating
                        if (now - last_action_time) < COMMAND_COOLDOWN_SEC:
                            continue
                        if last_cmd == best_label and (now - last_action_time) < (COMMAND_COOLDOWN_SEC * 1.5):
                            continue

                        # Print "during motion": we print right before and right after.
                        # (DJITelloPy calls are blocking; printing inside isn't feasible without threading.)
                        print(f"EXECUTE: {best_label}  ({best_conf*100:.1f}%)")

                        try:
                            do_drone_action(tello, best_label)
                        except Exception as e:
                            print(f"Drone command error for '{best_label}': {e}")

                        # After movement completes, print again (during/after motion context)
                        print(f"DONE: {best_label}  ({best_conf*100:.1f}%) - hovering")

                        last_cmd = best_label
                        last_action_time = now

    except KeyboardInterrupt:
        print("\nStopping (Ctrl+C).")
    finally:
        # Stop audio cleanly
        try:
            sd.stop()
        except Exception:
            pass

        # Safety: try to land on exit (comment out if you prefer manual)
        try:
            tello.land()
        except Exception:
            pass

        try:
            tello.end()
        except Exception:
            pass


if __name__ == "__main__":
    main()
