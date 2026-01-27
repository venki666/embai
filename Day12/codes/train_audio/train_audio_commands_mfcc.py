#!/usr/bin/env python3
"""
Train an audio command classifier using MFCC features from a folder-of-folders WAV dataset.

Expected dataset structure:
dataset/
  up/*.wav
  down/*.wav
  on/*.wav
  off/*.wav
  left/*.wav
  right/*.wav
  tello/*.wav
  _background_noise_/*.wav   (optional; used to generate "noise" samples)

Outputs:
- models/audio_cmd_mfcc.keras
- models/label_map.json
"""

import os
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Audio / DSP
import librosa

# ML
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# -----------------------
# Config
# -----------------------
DATASET_DIR = Path("dataset")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

COMMANDS = ["up", "down", "on", "off", "right", "left", "tello"]
BG_NOISE_DIRNAME = "_background_noise_"

SR = 8000
DURATION_SEC = 1.0
NUM_SAMPLES = int(SR * DURATION_SEC)

# MFCC params (tune if you like)
N_MFCC = 20
N_FFT = 256
HOP_LENGTH = 80   # 10 ms hop @ 8kHz -> 80 samples
WIN_LENGTH = 200  # 25 ms window @ 8kHz -> 200 samples

# Training
SEED = 42
TEST_SPLIT = 0.15
VAL_SPLIT = 0.15
BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 1e-3

# Noise/augmentation
ADD_NOISE_PROB = 0.35
NOISE_LABEL = "noise"  # extra class for background/noise
NOISE_SAMPLES_PER_EPOCH = 2000  # synthetic 1s crops from background noise wavs (approx)
SNR_DB_RANGE = (5, 25)  # mix noise with random SNR between 5 and 25 dB

# -----------------------
# Utilities
# -----------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def list_wavs(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    return sorted([p for p in folder.rglob("*.wav") if p.is_file()])

def load_wav_mono(path: Path, sr: int = SR) -> np.ndarray:
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y.astype(np.float32)

def fix_length(y: np.ndarray, length: int = NUM_SAMPLES) -> np.ndarray:
    if len(y) < length:
        return np.pad(y, (0, length - len(y)), mode="constant")
    if len(y) > length:
        return y[:length]
    return y

def random_crop_1s(y: np.ndarray, length: int = NUM_SAMPLES) -> np.ndarray:
    if len(y) <= length:
        return fix_length(y, length)
    start = np.random.randint(0, len(y) - length + 1)
    return y[start:start + length]

def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)) + 1e-12))

def mix_with_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Scale noise to achieve desired SNR and add to clean."""
    clean_rms = rms(clean)
    noise_rms = rms(noise)
    if noise_rms < 1e-8:
        return clean
    desired_noise_rms = clean_rms / (10 ** (snr_db / 20.0))
    noise_scaled = noise * (desired_noise_rms / noise_rms)
    mixed = clean + noise_scaled
    return np.clip(mixed, -1.0, 1.0)

def compute_mfcc(y: np.ndarray) -> np.ndarray:
    """Return MFCC array shaped (time, n_mfcc)."""
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=SR,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
    )  # shape: (n_mfcc, frames)
    mfcc = mfcc.astype(np.float32)
    mfcc = mfcc.T  # -> (frames, n_mfcc)
    return mfcc

def normalize_mfcc(mfcc: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (mfcc - mean) / (std + 1e-8)

def build_model(input_shape: Tuple[int, int, int], num_classes: int) -> keras.Model:
    """
    A compact CNN on MFCC "images": (frames, n_mfcc, 1)
    """
    inputs = keras.Input(shape=input_shape)

    x = layers.Conv2D(24, (3, 3), padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool2D((2, 2))(x)

    x = layers.Conv2D(48, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool2D((2, 2))(x)

    x = layers.Conv2D(96, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.25)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    return model

# -----------------------
# Dataset building
# -----------------------
def gather_files(dataset_dir: Path) -> Dict[str, List[Path]]:
    files = {}
    for label in COMMANDS:
        files[label] = list_wavs(dataset_dir / label)

    # background noise wavs
    bg_files = list_wavs(dataset_dir / BG_NOISE_DIRNAME)
    files[NOISE_LABEL] = bg_files  # store raw noise sources here
    return files

def create_examples(files: Dict[str, List[Path]]) -> Tuple[List[np.ndarray], List[str]]:
    """
    Load all command wavs into memory as 1s arrays.
    Also generate synthetic "noise" examples by cropping background noise wavs.
    """
    X_audio: List[np.ndarray] = []
    y_labels: List[str] = []

    # Commands (including tello)
    for label in COMMANDS:
        for wav_path in files[label]:
            y = load_wav_mono(wav_path)
            y = fix_length(y)
            X_audio.append(y)
            y_labels.append(label)

    # Synthetic noise examples from background noise (if present)
    bg_sources = files.get(NOISE_LABEL, [])
    if bg_sources:
        for _ in range(NOISE_SAMPLES_PER_EPOCH):
            src = random.choice(bg_sources)
            y = load_wav_mono(src)
            y = random_crop_1s(y)
            X_audio.append(y)
            y_labels.append(NOISE_LABEL)

    return X_audio, y_labels

def add_optional_noise_augment(
    audio: np.ndarray,
    bg_sources: List[Path],
    prob: float = ADD_NOISE_PROB
) -> np.ndarray:
    if (not bg_sources) or (np.random.rand() > prob):
        return audio
    src = random.choice(bg_sources)
    n = load_wav_mono(src)
    n = random_crop_1s(n)
    snr_db = np.random.uniform(*SNR_DB_RANGE)
    return mix_with_snr(audio, n, snr_db)

def split_indices(n: int, test_split: float, val_split: float, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)

    n_test = int(n * test_split)
    n_val = int(n * val_split)
    test_idx = idx[:n_test]
    val_idx = idx[n_test:n_test + n_val]
    train_idx = idx[n_test + n_val:]
    return train_idx, val_idx, test_idx

def compute_global_mfcc_norm(mfcc_list: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute mean/std over all MFCC frames (across all samples).
    """
    all_frames = np.concatenate(mfcc_list, axis=0)  # (sum_frames, n_mfcc)
    mean = np.mean(all_frames, axis=0, keepdims=True)
    std = np.std(all_frames, axis=0, keepdims=True)
    return mean.astype(np.float32), std.astype(np.float32)

def main():
    set_seed(SEED)

    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"Dataset folder not found: {DATASET_DIR.resolve()}")

    files = gather_files(DATASET_DIR)
    bg_sources = files.get(NOISE_LABEL, [])

    # Build examples (audio arrays + labels)
    X_audio, y_labels = create_examples(files)

    if len(X_audio) == 0:
        raise RuntimeError("No WAV files found. Check your dataset/ folder structure.")

    # Label map (commands + noise)
    labels_sorted = COMMANDS + [NOISE_LABEL]
    label_to_id = {lab: i for i, lab in enumerate(labels_sorted)}
    id_to_label = {i: lab for lab, i in label_to_id.items()}

    # Convert to MFCC features
    mfcc_list: List[np.ndarray] = []
    for audio in X_audio:
        audio_aug = add_optional_noise_augment(audio, bg_sources)
        mfcc = compute_mfcc(audio_aug)  # (frames, n_mfcc)
        mfcc_list.append(mfcc)

    # Ensure same #frames for all (should be consistent with fixed params)
    frame_counts = [m.shape[0] for m in mfcc_list]
    frames = int(np.median(frame_counts))

    # If any mismatch, pad/trim MFCC to 'frames'
    def fix_mfcc_len(m: np.ndarray, target_frames: int) -> np.ndarray:
        if m.shape[0] < target_frames:
            pad = np.zeros((target_frames - m.shape[0], m.shape[1]), dtype=np.float32)
            return np.vstack([m, pad])
        if m.shape[0] > target_frames:
            return m[:target_frames, :]
        return m

    mfcc_list = [fix_mfcc_len(m, frames) for m in mfcc_list]

    # Global normalization stats computed on *all* samples (simple + effective)
    mean, std = compute_global_mfcc_norm(mfcc_list)

    # Pack tensors: (N, frames, n_mfcc, 1)
    X = np.stack([normalize_mfcc(m, mean, std) for m in mfcc_list], axis=0).astype(np.float32)
    X = X[..., np.newaxis]
    y = np.array([label_to_id[l] for l in y_labels], dtype=np.int64)

    # Split
    train_idx, val_idx, test_idx = split_indices(len(X), TEST_SPLIT, VAL_SPLIT, SEED)
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    print(f"Total samples: {len(X)} | Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    print("Classes:", labels_sorted)

    # Build model
    model = build_model(input_shape=X_train.shape[1:], num_classes=len(labels_sorted))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        shuffle=True,
        verbose=1
    )

    # Evaluate
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest accuracy: {test_acc:.4f} | Test loss: {test_loss:.4f}")

    # Save model + metadata
    model_path = MODEL_DIR / "audio_cmd_mfcc.keras"
    model.save(model_path)

    meta = {
        "sr": SR,
        "duration_sec": DURATION_SEC,
        "num_samples": NUM_SAMPLES,
        "n_mfcc": N_MFCC,
        "n_fft": N_FFT,
        "hop_length": HOP_LENGTH,
        "win_length": WIN_LENGTH,
        "frames": frames,
        "labels": labels_sorted,
        "label_to_id": label_to_id,
        "id_to_label": id_to_label,
        "mfcc_mean": mean.squeeze(0).tolist(),
        "mfcc_std": std.squeeze(0).tolist(),
    }
    with open(MODEL_DIR / "label_map.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved model: {model_path}")
    print(f"Saved metadata: {MODEL_DIR / 'label_map.json'}")

if __name__ == "__main__":
    main()
