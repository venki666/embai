#include <M5Unified.h>
#include <SD.h>
#include <SPI.h>

// --- Configuration ---
#define PIN_SD_CS 4            // Core2 SD Card Chip Select Pin
#define SAMPLE_RATE 16000      // Audio Sample Rate (16kHz is standard for voice)
#define RECORD_BUFFER_SIZE 512 // Buffer size for reading audio

// --- Global Variables ---
File audioFile;
bool isRecording = false;
int16_t record_buffer[RECORD_BUFFER_SIZE];

// --- WAV Header Struct ---
struct WavHeader {
  char riff[4];           // "RIFF"
  uint32_t overall_size;  // file size - 8
  char wave[4];           // "WAVE"
  char fmt_chunk_marker[4]; // "fmt "
  uint32_t length_of_fmt; // 16
  uint16_t format_type;   // 1 (PCM)
  uint16_t channels;      // 1 (Mono)
  uint32_t sample_rate;   
  uint32_t byterate;      // SampleRate * NumChannels * BitsPerSample/8
  uint16_t block_align;   // NumChannels * BitsPerSample/8
  uint16_t bits_per_sample; // 16
  char data_chunk_header[4]; // "data"
  uint32_t data_size;     // NumSamples * NumChannels * BitsPerSample/8
};

// --- Helper: Write WAV Header ---
void writeWavHeader(File &file, int sampleRate) {
  WavHeader header;
  memcpy(header.riff, "RIFF", 4);
  header.overall_size = 0; // Placeholder, update later
  memcpy(header.wave, "WAVE", 4);
  memcpy(header.fmt_chunk_marker, "fmt ", 4);
  header.length_of_fmt = 16;
  header.format_type = 1; // PCM
  header.channels = 1;    // Mono
  header.sample_rate = sampleRate;
  header.bits_per_sample = 16;
  header.block_align = (header.channels * header.bits_per_sample) / 8;
  header.byterate = header.sample_rate * header.block_align;
  memcpy(header.data_chunk_header, "data", 4);
  header.data_size = 0;   // Placeholder, update later

  file.write((uint8_t*)&header, sizeof(WavHeader));
}

// --- Helper: Update WAV Header Sizes ---
void updateWavHeader(File &file) {
  uint32_t fileSize = file.size();
  uint32_t dataSize = fileSize - sizeof(WavHeader);
  uint32_t riffSize = fileSize - 8;

  file.seek(4); // Offset for 'overall_size'
  file.write((uint8_t*)&riffSize, 4);
  
  file.seek(40); // Offset for 'data_size'
  file.write((uint8_t*)&dataSize, 4);
  
  file.close();
  M5.Display.println("Header updated. Saved.");
}

void setup() {
  // 1. Init M5Unified
  auto cfg = M5.config();
  M5.begin(cfg);
  
  M5.Display.setTextSize(2);
  M5.Display.println("M5Core2 Audio Rec");
  M5.Display.println("Send 's' to toggle");

  // 2. Init Serial
  Serial.begin(115200);
  while (!Serial) delay(10);
  Serial.println("\n--- Serial Ready ---");

  // 3. Init SD Card
  // Core2 uses specific pins for SD, usually mapped correctly by default SPI, 
  // but we explicitly define CS pin 4.
  if (!SD.begin(PIN_SD_CS)) {
    M5.Display.setTextColor(RED);
    M5.Display.println("SD Init Failed!");
    Serial.println("SD Init Failed!");
    while (1); 
  }
  M5.Display.setTextColor(GREEN);
  M5.Display.println("SD OK");

  // 4. Init Microphone
  auto mic_cfg = M5.Mic.config();
  mic_cfg.sample_rate = SAMPLE_RATE;
  mic_cfg.task_priority = 2; // High priority for audio task
  M5.Mic.config(mic_cfg);
  
  if (!M5.Mic.begin()) {
    M5.Display.setTextColor(RED);
    M5.Display.println("Mic Init Failed!");
    while(1);
  }
  M5.Display.println("Mic OK. Waiting...");
}

void loop() {
  M5.update(); // Update button states/system

  // --- UART Command Handling ---
  if (Serial.available()) {
    char cmd = Serial.read();
    if (cmd == 's') {
      if (isRecording) {
        // STOP RECORDING
        isRecording = false;
        M5.Display.fillScreen(BLACK);
        M5.Display.setCursor(0,0);
        M5.Display.setTextColor(GREEN);
        M5.Display.println("Stopping...");
        Serial.println("Stopping recording...");
        
        // Finalize file
        updateWavHeader(audioFile);
        M5.Display.println("Done.");
        
      } else {
        // START RECORDING
        // Generate filename based on millis to avoid overwrite
        String fileName = "/rec_" + String(millis()) + ".wav";
        audioFile = SD.open(fileName, FILE_WRITE);
        
        if (audioFile) {
          isRecording = true;
          writeWavHeader(audioFile, SAMPLE_RATE);
          
          M5.Display.fillScreen(RED);
          M5.Display.setCursor(0,0);
          M5.Display.setTextColor(WHITE);
          M5.Display.printf("Recording to:\n%s\n", fileName.c_str());
          Serial.printf("Recording started: %s\n", fileName.c_str());
        } else {
          Serial.println("Failed to open file for writing");
        }
      }
    }
  }

  // --- Audio Loop ---
  if (isRecording) {
    // Read audio data from mic
    if (M5.Mic.record(record_buffer, RECORD_BUFFER_SIZE, SAMPLE_RATE)) {
      // Write to SD Card
      // We assume 16-bit mono audio (2 bytes per sample)
      audioFile.write((uint8_t*)record_buffer, RECORD_BUFFER_SIZE * 2);
    }
  }
}