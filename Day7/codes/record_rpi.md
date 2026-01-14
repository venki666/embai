### Pinouts for connecting INMP441 to RPi Zero 2W

Final 6-Pin Wiring Summary:

RPiZero2W        Pin                                   Purpose

VCC              Pin 1 (3.3V)	                            Power
GND            Pin 6 (GND)	                       Common Ground
L/R            Pin 9 or 14 (GND)	    Sets to Left Channel (Required for single mic)
SCK            Pin 12 (GPIO 18)	                   Serial Clock
WS             Pin 35 (GPIO 19)	      Word Select (Left/Right Clock)
SD            Pin 38 (GPIO 20)	                  Serial Data Output

### After connecting the Mic, type in the following command

```
arecord -l
```
### You should be seeing something like this if the Microphone is connected properly                                                   
**** List of CAPTURE Hardware Devices ****
card 0: sndrpigooglevoi [snd_rpi_googlevoicehat_soundcar], device 0: Google voiceHAT SoundCard HiFi voicehat-hifi-0 [Google voiceHAT SoundCard HiFi voicehat-hifi-0]
  Subdevices: 1/1
  Subdevice #0: subdevice #0

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
### Paste this command to record a audio for 5 seconds and store it in the home folder with .wav format
```
arecord -D plughw:0 -c1 -r 48000 -f S32_LE -t wav -d 5 -V mono summa.wav
```

