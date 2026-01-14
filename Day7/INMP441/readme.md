```
sudo apt install vim
```
```
sudo vim /boot/firmware/config.txt
```
### This is how the /boot/firmware/config.txt should look

```
# Uncomment some or all of these to enable the optional hardware interfaces
dtparam=i2c_arm=on
dtparam=i2s=on
dtparam=spi=on

# Enable audio (loads snd_bcm2835)
dtparam=audio=off
dtoverlay=googlevoicehat-soundcard
```
### Create a new virtual environment
```
python3 -m venv ~/audio_proj
source ~/audio_proj/bin/activate
```

### Install dependency 
```
pip install --upgrade pip setuptools wheel
pip install numpy
pip install sshkeyboard==2.3.1
pip install scipy
pip install --no-cache-dir --force-reinstall pyaudio==0.2.14
pip install matplotlib --prefer-binary
```

### Install System Dependencies
```
sudo apt install git 
sudo apt install htop -y
sudo ldconfig 
```
### Check library install
```
python -c "import pyaudio; print(pyaudio.__version__)"
chmod +x record.py
python record.py
python analyze_audio.py rec_key_1768431585_lp.wav
deactivate
ctrl+c
python visualize_audio.py rec_key_1768431585_lp.wav
```
