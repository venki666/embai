import os
import tarfile
import requests
import io
from tqdm import tqdm

# --- Configuration ---
# Official URL for Google Speech Commands v0.02
DATASET_URL = "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
OUTPUT_DIR = "google_speech_commands_filtered"

# The specific words you want to keep
TARGET_WORDS = ["on", "up", "down", "left", "right", "off", "_background_noise_"]


def download_and_extract(url, target_words, output_dir):
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"[INFO] Created directory: {output_dir}")

    print(f"[INFO] Downloading dataset from {url}...")

    # Stream the download to avoid memory issues (it's ~2.4GB)
    response = requests.get(url, stream=True)

    if response.status_code == 200:
        # Get total file size for progress bar
        total_size = int(response.headers.get('content-length', 0))

        # Load the stream into a TarFile object
        # We use BytesIO to wrap the raw stream so tarfile can read it
        # Note: This loads the archive into RAM or temp storage depending on implementation.
        # For strictly low-RAM systems, it's safer to download to a temp file first.
        # Below is a "Download to File then Extract" approach for stability.

        temp_filename = "speech_commands_temp.tar.gz"

        with open(temp_filename, 'wb') as f, tqdm(
                desc="Downloading",
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=1024):
                size = f.write(data)
                bar.update(size)

        print("\n[INFO] Download complete. Extracting specific folders...")

        # Open the downloaded tar file
        with tarfile.open(temp_filename, "r:gz") as tar:
            # Filter members
            members_to_extract = []
            for member in tar.getmembers():
                # The tar file structure is flat: "word/filename.wav"
                # We check if the file starts with one of our target words
                for word in target_words:
                    if member.name.startswith(word + "/"):
                        members_to_extract.append(member)
                        break

            # Extract only the matching members
            tar.extractall(path=output_dir, members=members_to_extract)
            print(f"[SUCCESS] Extracted {len(members_to_extract)} files to '{output_dir}'")

        # Cleanup temp file
        os.remove(temp_filename)
        print("[INFO] Temporary file removed.")

    else:
        print(f"[ERROR] Failed to download. Status code: {response.status_code}")


if __name__ == "__main__":
    # Note: Google uses "_background_noise_" with underscores in the tar file
    # We ensure our target list matches the folder names exactly.
    # Standard words: 'up', 'down' etc.
    # Noise folder: '_background_noise_'

    print(f"Targeting words: {TARGET_WORDS}")
    download_and_extract(DATASET_URL, TARGET_WORDS, OUTPUT_DIR)