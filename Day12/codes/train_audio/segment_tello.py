#!/usr/bin/env python3
"""
Segment repeated wake-word "tello" from a long WAV into 1-second WAV clips.

Approach (simple + robust for repeated single word):
1) Load audio, convert to mono, resample to target_sr (default 8 kHz)
2) Compute short-time RMS energy
3) Threshold RMS to get "active speech" mask
4) Convert mask to time-intervals, merge nearby intervals
5) For each interval, cut a 1.0s clip (centered on the interval), pad/trim to exactly 1 second
6) Save clips to an output folder

Works best if the file is mostly: silence + "tello" + silence + "tello" ...
If you have a lot of background noise, tune THRESH, MIN_ACTIVE_MS, and MERGE_GAP_MS.

Usage:
  python segment_tello.py --in_wav /mnt/data/tello.wav --out_dir tello_clips

Dependencies:
  pip install numpy librosa soundfile
"""

import argparse
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf


def fix_length(y: np.ndarray, target_len: int) -> np.ndarray:
    if y.size < target_len:
        return np.pad(y, (0, target_len - y.size), mode="constant")
    if y.size > target_len:
        return y[:target_len]
    return y


def rms_activity_mask(
    y: np.ndarray,
    sr: int,
    frame_ms: float,
    hop_ms: float,
    thresh: float,
) -> tuple[np.ndarray, int, int]:
    frame_len = max(1, int(sr * frame_ms / 1000.0))
    hop_len = max(1, int(sr * hop_ms / 1000.0))

    rms = librosa.feature.rms(y=y, frame_length=frame_len, hop_length=hop_len, center=True)[0]
    # Normalize RMS to reduce dependency on recording gain
    rms_norm = rms / (np.max(rms) + 1e-12)

    mask = rms_norm >= thresh
    return mask, frame_len, hop_len


def mask_to_intervals(mask: np.ndarray, hop_len: int, sr: int) -> list[tuple[int, int]]:
    """Convert boolean mask (per frame) to sample intervals [start, end)."""
    if mask.size == 0:
        return []

    # Find rising/falling edges
    m = mask.astype(np.int32)
    diff = np.diff(np.concatenate([[0], m, [0]]))
    starts = np.where(diff == 1)[0]  # frame indices
    ends = np.where(diff == -1)[0]   # frame indices

    intervals = []
    for s_f, e_f in zip(starts, ends):
        s = int(s_f * hop_len)
        e = int(e_f * hop_len)
        intervals.append((s, e))
    return intervals


def merge_intervals(intervals: list[tuple[int, int]], merge_gap: int) -> list[tuple[int, int]]:
    if not intervals:
        return []

    intervals = sorted(intervals, key=lambda x: x[0])
    merged = [intervals[0]]

    for s, e in intervals[1:]:
        ps, pe = merged[-1]
        if s <= pe + merge_gap:
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))
    return merged


def filter_intervals(intervals: list[tuple[int, int]], min_len: int) -> list[tuple[int, int]]:
    return [(s, e) for s, e in intervals if (e - s) >= min_len]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_wav", type=str, required=True, help="Input long WAV file")
    ap.add_argument("--out_dir", type=str, default="tello_clips", help="Output directory for 1s clips")
    ap.add_argument("--target_sr", type=int, default=8000, help="Resample audio to this rate")
    ap.add_argument("--clip_sec", type=float, default=1.0, help="Clip duration (seconds)")
    ap.add_argument("--frame_ms", type=float, default=25.0, help="RMS frame length in ms")
    ap.add_argument("--hop_ms", type=float, default=10.0, help="RMS hop length in ms")

    # Thresholding / cleanup knobs
    ap.add_argument("--thresh", type=float, default=0.12, help="Normalized RMS threshold (0..1)")
    ap.add_argument("--min_active_ms", type=float, default=150.0, help="Minimum active interval to keep (ms)")
    ap.add_argument("--merge_gap_ms", type=float, default=120.0, help="Merge intervals if gap is <= this (ms)")
    ap.add_argument("--pad_ms", type=float, default=80.0, help="Extra padding around detected interval (ms) before centering")

    args = ap.parse_args()

    in_wav = Path(args.in_wav)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load + resample
    y, sr = librosa.load(str(in_wav), sr=args.target_sr, mono=True)
    y = y.astype(np.float32)

    clip_len = int(args.target_sr * args.clip_sec)
    min_len = int(args.target_sr * args.min_active_ms / 1000.0)
    merge_gap = int(args.target_sr * args.merge_gap_ms / 1000.0)
    pad = int(args.target_sr * args.pad_ms / 1000.0)

    # Activity mask -> intervals
    mask, frame_len, hop_len = rms_activity_mask(
        y=y,
        sr=args.target_sr,
        frame_ms=args.frame_ms,
        hop_ms=args.hop_ms,
        thresh=args.thresh,
    )
    intervals = mask_to_intervals(mask, hop_len=hop_len, sr=args.target_sr)
    intervals = merge_intervals(intervals, merge_gap=merge_gap)
    intervals = filter_intervals(intervals, min_len=min_len)

    if not intervals:
        print("No segments detected. Try lowering --thresh or --min_active_ms.")
        return

    # Create 1-second clips centered on each detected interval
    saved = 0
    for i, (s, e) in enumerate(intervals):
        # Expand interval slightly (helps include full word)
        s2 = max(0, s - pad)
        e2 = min(y.size, e + pad)

        # Center a 1-second window around the expanded interval midpoint
        mid = (s2 + e2) // 2
        start = int(mid - clip_len // 2)
        end = start + clip_len

        # Shift window to fit inside audio bounds
        if start < 0:
            start = 0
            end = clip_len
        if end > y.size:
            end = y.size
            start = max(0, end - clip_len)

        clip = y[start:end]
        clip = fix_length(clip, clip_len)

        out_path = out_dir / f"tello_{saved:04d}.wav"
        sf.write(str(out_path), clip, args.target_sr, subtype="PCM_16")
        saved += 1

    print(f"Detected intervals: {len(intervals)}")
    print(f"Saved {saved} clips to: {out_dir.resolve()}")
    print("Tip: If clips start too early/late, adjust --pad_ms or --thresh.")


if __name__ == "__main__":
    main()
