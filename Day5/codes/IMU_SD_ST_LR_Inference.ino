#include <M5Unified.h>
#include <MadgwickAHRS.h> 
#include <SD.h>

// --- CONFIGURATION ---
Madgwick filter;
const float SENSOR_RATE = 100.0f; 
unsigned long last_update = 0;
bool is_recording = false;

// --- SENSOR VARIABLES ---
float ax, ay, az;
float gx, gy, gz;
float ax_s = 0, ay_s = 0, az_s = 0; // Smoothed Accel
float gx_s = 0, gy_s = 0, gz_s = 0; // Smoothed Gyro
const float ALPHA = 0.2;            // Smoothing Factor

// --- FUSION VARIABLES (Ground Truth) ---
float roll, pitch, yaw;

// --- LINEAR REGRESSION MODEL (From Python) ---
// Formula: y = (w1 * ax) + (w2 * ay) + (w3 * az) + intercept

// Roll Coefficients
const float R_W_AX = -7.79178;
const float R_W_AY = 59.29750;
const float R_W_AZ = -7.94485;
const float R_Bias = 10.57404;

// Pitch Coefficients
const float P_W_AX = -63.72033;
const float P_W_AY = 0.72477;
const float P_W_AZ = 11.50674;
const float P_Bias = -15.72815;

// LR Output Variables
float roll_lr = 0;
float pitch_lr = 0;

File logFile;

void setup() {
  auto cfg = M5.config();
  cfg.serial_baudrate = 115200;
  cfg.internal_spk = false; 
  cfg.internal_mic = false; 
  M5.begin(cfg);
  
  M5.Imu.begin();
  filter.begin(SENSOR_RATE);

  // Init SD
  if (!SD.begin(GPIO_NUM_4, SPI, 25000000)) { 
    M5.Display.println("SD Failed!");
  }

  // Setup Screen Layout
  M5.Display.setTextSize(2);
  M5.Display.fillScreen(BLACK);
  
  // Static Labels
  M5.Display.setCursor(10, 10);
  M5.Display.setTextColor(CYAN); M5.Display.print("FUSION");
  M5.Display.setCursor(160, 10);
  M5.Display.setTextColor(ORANGE); M5.Display.print("LIN.REG");
  
  M5.Display.drawLine(155, 0, 155, 240, WHITE); // Vertical Separator
  M5.Display.drawLine(0, 40, 320, 40, WHITE);   // Header Separator

  Serial.println("timestamp,roll_fusion,pitch_fusion,roll_lr,pitch_lr");
}

void loop() {
  M5.update();

  // --- UART RECORDING TRIGGER ---
  if (Serial.available()) {
    char cmd = Serial.read();
    if (cmd == 's' || cmd == 'S') { 
      is_recording = !is_recording;
      // Note: We aren't redrawing the full screen here to keep the UI clean
      // You can add a small indicator icon if needed.
    }
  }
  
  // --- 100Hz LOOP ---
  if (millis() - last_update >= (1000.0 / SENSOR_RATE)) {
    last_update = millis();
    unsigned long ts = millis();

    // 1. Get Raw Data
    M5.Imu.getAccelData(&ax, &ay, &az);
    M5.Imu.getGyroData(&gx, &gy, &gz);

    // 2. Smooth Data (Low Pass Filter)
    ax_s = (ALPHA * ax) + ((1.0 - ALPHA) * ax_s);
    ay_s = (ALPHA * ay) + ((1.0 - ALPHA) * ay_s);
    az_s = (ALPHA * az) + ((1.0 - ALPHA) * az_s);
    
    gx_s = (ALPHA * gx) + ((1.0 - ALPHA) * gx_s);
    gy_s = (ALPHA * gy) + ((1.0 - ALPHA) * gy_s);
    gz_s = (ALPHA * gz) + ((1.0 - ALPHA) * gz_s);

    // 3. Fusion Algorithm (Ground Truth)
    filter.updateIMU(gx, gy, gz, ax, ay, az);
    roll  = filter.getRoll();
    pitch = filter.getPitch();
    yaw   = filter.getYaw();

    // 4. LINEAR REGRESSION INFERENCE (The AI Part)
    // Roll Prediction
    roll_lr = (R_W_AX * ax_s) + (R_W_AY * ay_s) + (R_W_AZ * az_s) + R_Bias;
    
    // Pitch Prediction
    pitch_lr = (P_W_AX * ax_s) + (P_W_AY * ay_s) + (P_W_AZ * az_s) + P_Bias;

    // 5. UPDATE DISPLAY (Every 100ms approx to avoid flicker, or use sprite)
    // Using simple overwrite for code clarity
    static unsigned long last_draw = 0;
    if (millis() - last_draw > 100) { 
        last_draw = millis();
        
        // Clear previous values area (Partial clear is faster)
        M5.Display.fillRect(0, 50, 320, 100, BLACK);
        
        // -- FUSION COLUMN (Left) --
        M5.Display.setTextColor(CYAN);
        M5.Display.setCursor(10, 60);
        M5.Display.printf("R: %.1f", roll);
        M5.Display.setCursor(10, 90);
        M5.Display.printf("P: %.1f", pitch);

        // -- LIN.REG COLUMN (Right) --
        M5.Display.setTextColor(ORANGE);
        M5.Display.setCursor(170, 60);
        M5.Display.printf("R: %.1f", roll_lr);
        M5.Display.setCursor(170, 90);
        M5.Display.printf("P: %.1f", pitch_lr);
        
        // Recording Status at bottom
        M5.Display.setCursor(10, 200);
        M5.Display.setTextColor(is_recording ? GREEN : RED);
        M5.Display.print(is_recording ? "REC: ON " : "REC: OFF");
    }

    // 6. SERIAL STREAM
    Serial.printf("%lu,%.2f,%.2f,%.2f,%.2f\n", ts, roll, pitch, roll_lr, pitch_lr);

    // 7. LOGGING
    if (is_recording && logFile) {
       // Re-open if closed logic or keep open strategy
       logFile = SD.open("/lr_log.csv", FILE_APPEND);
       if (logFile) {
          logFile.printf("%lu,%.2f,%.2f,%.2f,%.2f\n", ts, roll, pitch, roll_lr, pitch_lr);
          logFile.close();
       }
    }
  }
}