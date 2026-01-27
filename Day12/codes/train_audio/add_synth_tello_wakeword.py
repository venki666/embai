import os
from gtts import gTTS
from pydub import AudioSegment
import numpy as np

# --- Configuration ---
WAKE_WORD = "Tello"
OUTPUT_DIR = "dataset/tello"
SAMPLE_RATE = 8000  # Must match your training script
NUM_SAMPLES = 100   # Number of synthetic variations

def generate_synthetic_tello():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"[INFO] Created directory: {OUTPUT_DIR}")

    print(f"[INFO] Generating {NUM_SAMPLES} synthetic samples for '{WAKE_WORD}'...")

    # 1. Generate the base audio using TTS
    tts = gTTS(text=WAKE_WORD, lang='en', slow=False)
    temp_mp3 = "temp_tello.mp3"
    tts.save(temp_mp3)

    # 2. Load with pydub and convert to Mono, 8kHz
    base_audio = AudioSegment.from_mp3(temp_mp3)
    base_audio = base_audio.set_frame_rate(SAMPLE_RATE).set_channels(1)

    for i in range(NUM_SAMPLES):
        # Vary the speed slightly (0.9x to 1.1x) to create diversity
        speed_change = np.random.uniform(0.9, 1.1)
        new_sample = base_audio._spawn(base_audio.raw_data, overrides={
            "frame_rate": int(base_audio.frame_rate * speed_change)
        }).set_frame_rate(SAMPLE_RATE)

        # Pad or truncate to exactly 1 second (8000 samples)
        target_ms = 1000
        if len(new_sample) < target_ms:
            new_sample = new_sample + AudioSegment.silent(duration=target_ms - len(new_sample))
        else:
            new_sample = new_sample[:target_ms]

        # Save as WAV
        file_name = f"tello_synth_{i:03d}.wav"
        new_sample.export(os.path.join(OUTPUT_DIR, file_name), format="wav")

    os.remove(temp_mp3)
    print(f"[SUCCESS] Generated {NUM_SAMPLES} samples in {OUTPUT_DIR}")

if __name__ == "__main__":
    generate_synthetic_tello()