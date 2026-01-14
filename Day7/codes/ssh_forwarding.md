## Configure the Raspberry Pi (The Server)
Since your Pi is likely running Raspberry Pi OS Lite (no desktop), it is missing a few critical components required for X11 authentication.

**Step1: SSH into your Pi:**
```
ssh username@raspberrypi.local
```
Install the X Authorization Package: By default, the "Lite" OS does not include xauth, which is required for X11 forwarding to authorize the display connection.
```
sudo apt update
sudo apt install xauth
```
Install X11 Apps for Testing (Optional but Recommended): Install a lightweight app like xeyes or xclock to verify the connection without loading a heavy program.
```
sudo apt install x11-apps
```
Enable X11 Forwarding in SSH Config: Open the SSH daemon configuration file:

```
sudo nano /etc/ssh/sshd_config
```
Find the line X11Forwarding and ensure it is uncommented and set to yes. If it doesn't exist, add it:
```
X11Forwarding yes
```
(Note: You generally do not need to change X11UseLocalhost. Default is usually fine.)

Restart the SSH Service:
```
sudo systemctl restart ssh
```

### Step 2: Configure the Client (The Display)
The configuration depends on your computer's operating system.

**Option A: Windows**
Windows does not have a built-in X Server to display the windows. You must install one.

Install an X Server:

Download and install VcXsrv (recommended) or Xming.

Run VcXsrv (XLaunch):

Accept defaults (Multiple windows).

Important: On the "Extra settings" page, ensure "Disable access control" is checked. (This prevents permission errors when the Pi tries to connect back to Windows).

Connect via PowerShell / Command Prompt: If you have Windows 10/11, you can use the built-in SSH client.

**PowerShell**
```
ssh -X username@raspberrypi.local
```
(Note: If -X is slow, try -Y which is "Trusted X11 forwarding" and sometimes faster).

**Connect via PuTTY (Alternative):**
Go to Connection > SSH > X11.

**Check Enable X11 forwarding.**
X display location: localhost:0

**Option B: macOS**
Macs used to have X11 built-in, but now you need a helper.

**Install XQuartz:**
Download and install XQuartz.

Log out and back in (or restart) your Mac to finalize the installation.

**Connect via Terminal:**
```
ssh -X username@raspberrypi.local
```

### Step 3: Verify and Run
Check the Display Variable: Once you are logged into the Pi via the SSH session (with the -X flag), run:

```
echo $DISPLAY
```
Success: You should see output like localhost:10.0.

Failure: If the line is empty, X11 forwarding was refused (check xauth installation or sshd_config on the Pi).

Launch a Test App: Run the test application you installed earlier:
```
xeyes
```
Result: A pair of eyes should appear on your computer's desktop that follow your mouse cursor.

Run Your Python Script (Example): If you are plotting a graph using Python (e.g., Matplotlib), ensure your code uses a backend that supports X11 (like TkAgg) or just let it auto-detect.
```
python3 my_plot_script.py
```
