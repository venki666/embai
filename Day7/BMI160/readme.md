```
sudo i2cdetect -y 1
```
```
python3 -m venv ~/imu_env
```
```
source ~/imu_env/bin/activate
```
```
pip install --upgrade pip
```
```
pip install smbus2 BMI160-i2c
```
```
pip install numpy ahrs
```
```
pip install pandas
```
```
pip install matplotlib
```
```
vim bmiscreen.py
```
```
python bmiscreen.py
```
```
cd imu_logs/
```
```
source ~/imu_env/bin/activate
```
```
vim bmi160_screen_fusion.py
```
```
source ~/imu_env/bin/activate
```
```
python bmi160_screen_fusion.py
```
```
vim analyzebmi160.py
```
```
source ~/imu_env/bin/activate
```
```
python analyzebmi160.py ~/imu_logs/bmi160_20260113_150737.csv
```
