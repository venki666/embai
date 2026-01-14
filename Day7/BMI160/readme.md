### Check for your i2c connection
```
sudo i2cdetect -y 1
```
### Create a storage directory for im logs
```
mkdir -p imu_logs
cd ..
```
### Create a virtual environment
```
python3 -m venv ~/imu_env
source ~/imu_env/bin/activate
```
### Update libraries
```
pip install --upgrade pip
pip install smbus2 BMI160-i2c
pip install numpy ahrs
pip install pandas
pip install matplotlib --prefer-binary
```

### Data Collection
Place your codes in the imu_logs directory
```
cd imu_logs/
python bmi160_screen_fusion.py
```
### Record the data
- Press Enter for recording the data
- Press Enter again to stop recording
- Type q to quit data collection

### Perform data analysis
```
python analyzebmi160.py ~/imu_logs/bmi160_20260113_150737.csv
```

### close the virtual environment
```
deactivate
```
