#include <M5Unified.h>
#include <SD.h>

// --- CONFIGURATION ---
const int SD_CS_PIN = GPIO_NUM_4;  // Chip Select for M5Core2 SD Card
const int SAMPLE_RATE_HZ = 100;    // Logging Frequency (100Hz)
const char* FILENAME = "/raw_imu_log.csv";

// --- VARIABLES ---
bool isRecording = false;
unsigned long lastLoopTime = 0;
unsigned long loopInterval = 1000 / SAMPLE_RATE_HZ;

float ax, ay, az;
float gx, gy, gz;

void setup() {
  // 1. Initialize M5Stack (Configures I2C, Screen, Power, etc.)
  auto cfg = M5.config();
  cfg.serial_baudrate = 115200; 
  M5.begin(cfg);

  // 2. Initialize IMU
  M5.Imu.begin();

  // 3. Initialize SD Card
  // M5Core2 uses GPIO 4 for SD CS. 
  // We check if it mounts successfully.
  if (!SD.begin(SD_CS_PIN, SPI, 25000000)) {
    M5.Display.fillScreen(TFT_RED);
    M5.Display.setTextSize(2);
    M5.Display.setCursor(10, 100);
    M5.Display.println("SD Card Failed!");
    M5.Display.println("Insert SD & Reset");
    while (1); // Halt if no SD
  }

  // 4. Setup Display
  M5.Display.fillScreen(TFT_BLACK);
  M5.Display.setTextSize(3);
  M5.Display.setTextColor(TFT_WHITE, TFT_BLACK);
  M5.Display.setCursor(10, 10);
  M5.Display.println("IMU LOGGER");
  
  updateStatusScreen();

  // 5. Print CSV Header to Serial
  // Allows tools like Serial Plotter to identify columns
  Serial.println("timestamp,ax,ay,az,gx,gy,gz");
}

void loop() {
  M5.update(); // Update button states (handled by M5Unified internally)

  // --- 1. CHECK FOR UART COMMAND 's' ---
  if (Serial.available()) {
    char cmd = Serial.read();
    if (cmd == 's' || cmd == 'S') {
      isRecording = !isRecording;
      updateStatusScreen();
      
      // Optional: Add a separator line in SD file to mark new session
      if (isRecording) {
        File logFile = SD.open(FILENAME, FILE_APPEND);
        if (logFile) {
          logFile.println("--- NEW SESSION ---");
          logFile.close();
        }
      }
    }
  }

  // --- 2. SAMPLING LOOP (100Hz) ---
  if (millis() - lastLoopTime >= loopInterval) {
    lastLoopTime = millis();
    unsigned long timestamp = millis();

    // Read Raw Data
    M5.Imu.getAccelData(&ax, &ay, &az);
    M5.Imu.getGyroData(&gx, &gy, &gz);

    // --- 3. SERIAL OUTPUT (Always Stream) ---
    // Format: timestamp, ax, ay, az, gx, gy, gz
    Serial.printf("%lu,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n", 
                  timestamp, ax, ay, az, gx, gy, gz);

    // --- 4. SD CARD LOGGING (Only if Recording) ---
    if (isRecording) {
      File logFile = SD.open(FILENAME, FILE_APPEND);
      if (logFile) {
        logFile.printf("%lu,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n", 
                       timestamp, ax, ay, az, gx, gy, gz);
        logFile.close();
      } else {
        // Simple error indication on screen corner if write fails
        M5.Display.fillCircle(310, 10, 5, TFT_RED); 
      }
    }
  }
}

// Helper to update the UI
void updateStatusScreen() {
  M5.Display.fillRect(0, 100, 320, 100, TFT_BLACK); // Clear area
  M5.Display.setCursor(20, 110);
  M5.Display.setTextSize(3);
  
  if (isRecording) {
    M5.Display.setTextColor(TFT_RED, TFT_BLACK);
    M5.Display.println("RECORDING...");
    M5.Display.setTextSize(2);
    M5.Display.println("(Send 's' to Stop)");
  } else {
    M5.Display.setTextColor(TFT_GREEN, TFT_BLACK);
    M5.Display.println("IDLE");
    M5.Display.setTextSize(2);
    M5.Display.println("(Send 's' to Start)");
  }
}