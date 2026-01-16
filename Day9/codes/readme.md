Running TensorFlow Lite (.tflite) on a Raspberry Pi Zero 2 W (32-bit OS) requires a slightly different setup than a standard desktop because you cannot simply install the full tensorflow library (it is too heavy and often not supported on 32-bit armv7l architecture).

Instead, you must use the lightweight tflite-runtime.

1. Hardware & OS Check
Ensure you are running the 32-bit version of Raspberry Pi OS (Legacy/Buster or Bullseye). Run this command to confirm your architecture:
```
uname -m
```
# Output should be 'armv7l' (which indicates 32-bit userspace)
2. Installation Instructions
Run the following commands on your Raspberry Pi Zero 2 W terminal to install the necessary audio drivers and the TFLite runtime.

Step A: System Dependencies
```
sudo apt-get update
sudo apt-get install python3-pip python3-numpy libatlas-base-dev portaudio19-dev libsndfile1
```
Step B: Install TFLite Runtime (32-bit) The standard pip repo sometimes misses the specific wheel for Pi Zero. Use the Google Coral repository for a reliable build:
```
pip3 install tflite-runtime
```
If the command above fails, try forcing the extra index:
```
pip3 install --extra-index-url https://google-coral.github.io/py-repo/ tflite-runtime
```
Step C: Install Audio Libraries Note: librosa can be very slow to install on a Pi Zero because it compiles heavy dependencies. Be patient, or consider using python_speech_features if you only need MFCCs.
```
pip3 install sounddevice librosa scipy
```
