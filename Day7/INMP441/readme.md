```
sudo apt install git -y
```
```
sudo apt install htop -y
```
```
python3 -m venv ~/audio_proj
```
```
source ~/audio_proj/bin/activate
```
```
pip install --upgrade pip setuptools wheel
```
```
pip install numpy
```
```
pip install sshkeyboard==2.3.1
```
```
pip install scipy
```
```
pip install --no-cache-dir --force-reinstall pyaudio==0.2.14
```
```
python -c "import pyaudio; print(pyaudio.__version__)"
```
```
sudo apt install libportaudio2 portaudio19-dev -y
```
```
sudo ldconfig  # Updates library cache
```
```
source ~/audio_proj/bin/activate
```
```
vim record.py
```
```
chmod +x record.py
```
```
python record.py
```
```
python analyze_audio.py rec_key_1768431585_lp.wav
```
