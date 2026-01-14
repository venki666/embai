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
```

### Install System Dependencies
```
sudo apt update && sudo apt install libportaudio2 portaudio19-dev libatlas-base-dev python3-dev build-essential -y
sudo apt install git 
sudo apt install htop -y
sudo ldconfig 
```
### Check library install
```
python -c "import pyaudio; print(pyaudio.__version__)"
chmod +x record.py
python record.py
```
