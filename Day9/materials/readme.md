
###Prerequisites for Raspberry Pi Zero 2 W:

Hardware: RPi Zero 2 W, USB Microphone (or HAT), Power Supply.

OS: Raspberry Pi OS (Legacy/Buster is often more stable for audio, but Bullseye/Bookworm works with PipeWire/PulseAudio configuration).

Libraries: You need to install the following. Note that installing full TensorFlow on a Pi Zero can be heavy; tflite-runtime is highly recommended for inference.
```
sudo apt-get update
sudo apt-get install python3-pyaudio libatlas-base-dev
pip3 install sounddevice numpy librosa
```
###For inference only (lighter than full tensorflow):
```
pip3 install tflite-runtime
```
(Note: If you stick with full TensorFlow for simplicity, pip3 install tensorflow works but is large. The code below uses full TensorFlow for compatibility with the previous .h5 files, but I will include a note on TFLite).

**Microphone Setup:**
Before running Python, ensure your USB mic is default. Run arecord -l to find your mic card number. Create/edit ~/.asoundrc if needed to set the default capture device.
