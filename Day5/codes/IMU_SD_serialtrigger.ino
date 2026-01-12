#include <M5Unified.h>
#include <MadgwickAHRS.h> 
#include <SD.h>

// --- CONFIGURATION ---
Madgwick filter;
const float SENSOR_RATE = 100.0f; // 100Hz Sampling
unsigned long last_update = 0;
bool is_recording = false;

// --- VARIABLES ---
float ax, ay, az;
float gx, gy, gz;
float ax_smooth = 0, ay_smooth = 0, az_smooth = 0;
const float ALPHA = 0.2; // Smoothing factor
float roll, pitch, yaw;

File logFile;

void setup() {
  auto cfg = M5.config();
  M5.begin(cfg);
  
  // 1. Init Serial
  Serial.begin(115200); 
  
  // 2. Init IMU & Filter
  M5.Imu.begin();
  filter.begin(SENSOR_RATE);

  // 3. Init SD Card
  if (!SD.begin(GPIO_NUM_4, SPI, 25000000)) { 
    M5.Display.println("SD Failed!");
  } else {
    M5.Display.println("SD Ready.");
  }
  
  // 4. UI Setup
  M5.Display.setTextSize(2);
  M5.Display.setCursor(0, 40);
  M5.Display.println("Send 's' to Record");

  // 5. PRINT CSV HEADER (Crucial for Visualization)
  // This tells the visualizer/logger what the columns are
  Serial.println("timestamp,ax_raw,ay_raw,az_raw,ax_smooth,ay_smooth,az_smooth,roll,pitch,yaw");
}

void loop() {
  M5.update();

  // --- UART TRIGGER LOGIC ---
  if (Serial.available()) {
    char cmd = Serial.read();
    if (cmd == 's' || cmd == 'S') { // Toggle on 's'
      is_recording = !is_recording;
      
      // Visual Feedback on Screen
      if (is_recording) {
        M5.Display.fillScreen(GREEN);
        M5.Display.setCursor(10, 50);
        M5.Display.setTextColor(BLACK);
        M5.Display.print("RECORDING TO SD...");
        
        // Open/Create File
        logFile = SD.open("/imu_log.csv", FILE_APPEND);
        if (!logFile) {
           logFile = SD.open("/imu_log.csv", FILE_WRITE);
           // Write header to file if it's new
           logFile.println("timestamp,ax_raw,ay_raw,az_raw,ax_smooth,ay_smooth,az_smooth,roll,pitch,yaw"); 
        }
      } else {
        M5.Display.fillScreen(BLACK);
        M5.Display.setCursor(10, 50);
        M5.Display.setTextColor(WHITE);
        M5.Display.print("STOPPED");
        M5.Display.setCursor(10, 80);
        M5.Display.print("Send 's' to start");
        
        if (logFile) logFile.close();
      }
    }
  }
  
  // --- SENSOR LOOP (100Hz) ---
  if (millis() - last_update >= (1000.0 / SENSOR_RATE)) {
    last_update = millis();
    unsigned long ts = millis();

    // 1. Get Data
    M5.Imu.getAccelData(&ax, &ay, &az);
    M5.Imu.getGyroData(&gx, &gy, &gz);

    // 2. Filter (Smoothing)
    ax_smooth = (ALPHA * ax) + ((1.0 - ALPHA) * ax_smooth);
    ay_smooth = (ALPHA * ay) + ((1.0 - ALPHA) * ay_smooth);
    az_smooth = (ALPHA * az) + ((1.0 - ALPHA) * az_smooth);

    // 3. Fusion (Madgwick)
    filter.updateIMU(gx, gy, gz, ax, ay, az);
    roll  = filter.getRoll();
    pitch = filter.getPitch();
    yaw   = filter.getYaw();

    // 4. STREAM ENTIRE DATA TO SERIAL (CSV Format)
    // format: timestamp, ax, ay, az, ax_s, ay_s, az_s, roll, pitch, yaw
    Serial.printf("%lu,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n", 
                  ts, ax, ay, az, ax_smooth, ay_smooth, az_smooth, roll, pitch, yaw);

    // 5. LOG TO SD (If Active)
    if (is_recording && logFile) {
      logFile.printf("%lu,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n", 
                     ts, ax, ay, az, ax_smooth, ay_smooth, az_smooth, roll, pitch, yaw);
    }
  }
}