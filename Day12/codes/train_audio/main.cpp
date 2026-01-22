#include <M5Unified.h>
#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

// --- Import Model Data ---
// Copy the content of model_data.cc here
extern const unsigned char model_tflite[];
extern const int model_tflite_len;

// --- Configuration ---
#define SAMPLE_RATE 8000
#define INPUT_LENGTH 8000
#define RECORD_TIME 1000 // 1 second

// TFLite Globals
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
tflite::ErrorReporter* error_reporter = nullptr;
constexpr int kTensorArenaSize = 40 * 1024; // Adjust based on model size
uint8_t tensor_arena[kTensorArenaSize];

// Audio Buffer
int16_t raw_audio[INPUT_LENGTH]; // Buffer for recording
int8_t *model_input_buffer = nullptr;

// Commands (Must match the Python order alphabetical usually!)
const char* WORDS[] = {"down", "land", "left", "right", "takeoff", "up"};

void setup() {
    auto cfg = M5.config();
    M5.begin(cfg);
    M5.Log.setLogLevel(m5::log_target_serial, ESP_LOG_INFO);

    M5.Display.setTextSize(2);
    M5.Display.print("Init TFLite...");

    // 1. Initialize Microphone
    auto mic_cfg = M5.Mic.config();
    mic_cfg.sample_rate = SAMPLE_RATE;
    mic_cfg.noise_filter_level = 0; // Set higher if noisy
    M5.Mic.config(mic_cfg);
    M5.Mic.begin();

    // 2. Initialize TFLite
    model = tflite::GetModel(model_tflite);
    static tflite::MicroMutableOpResolver<10> resolver;
    resolver.AddConv2D();
    resolver.AddReshape();
    resolver.AddFullyConnected();
    resolver.AddSoftmax();
    resolver.AddRelu();
    resolver.AddQuantize();
    resolver.AddDequantize();
    // Add other ops if your model uses them (check standard TFLM errors)

    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, kTensorArenaSize, nullptr);
    interpreter = &static_interpreter;

    interpreter->AllocateTensors();
    model_input_buffer = interpreter->input(0)->data.int8;

    M5.Display.println("Ready!");
}

void loop() {
    M5.update();

    // Simple Threshold Trigger (Voice Activity Detection)
    // Only record if volume is loud enough to save battery/cpu
    // Or just loop continuously for this demo:

    if (M5.Mic.isEnabled()) {
        // Record 1 second (Blocking for simplicity)
        // M5Unified Mic record fills the buffer
        if (M5.Mic.record(raw_audio, INPUT_LENGTH, SAMPLE_RATE)) {
             // Wait for recording to complete
             while (M5.Mic.isRecording()) {
                 delay(1);
             }

             // 3. Preprocess: Convert int16 audio to int8 for the model
             // The model expects int8.
             // int16 range: -32768 to 32767
             // int8 range: -128 to 127
             for (int i = 0; i < INPUT_LENGTH; i++) {
                 model_input_buffer[i] = (int8_t)(raw_audio[i] >> 8);
             }

             // 4. Inference
             TfLiteStatus invoke_status = interpreter->Invoke();
             if (invoke_status != kTfLiteOk) {
                 M5.Display.println("Invoke failed!");
                 return;
             }

             // 5. Output Processing
             int8_t* output = interpreter->output(0)->data.int8;

             // Find max probability
             int max_idx = 0;
             int8_t max_val = -128;
             for (int i = 0; i < 6; i++) { // 6 classes
                 if (output[i] > max_val) {
                     max_val = output[i];
                     max_idx = i;
                 }
             }

             // Threshold check (approx 0.6 confidence in int8 scale)
             // -128 to 127 mapping: 0 is approx 50%.
             if (max_val > 0) {
                 M5.Display.fillScreen(BLACK);
                 M5.Display.setCursor(10, 50);
                 M5.Display.printf("CMD: %s", WORDS[max_idx]);
                 M5.Display.printf("\nConf: %d", max_val);

                 // TODO: Insert UDP Send Code here to control Tello
             }
        }
    }
    delay(100);
}