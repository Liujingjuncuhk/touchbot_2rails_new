#include <Arduino.h>
#include <WiFi.h>
#include <WiFiUdp.h>
#include "FastAccelStepper.h"
#include <SCServo.h>

// --- WiFi & UDP Setup ---
const char* ssid = "jingjun_ubuntu";
const char* password = "12345678";

WiFiUDP udp;
unsigned int localUdpPort = 5005; // ESP32 监听的端口号
char incomingPacket[255];         // 存放接收数据的缓冲区

// Variables to store the PC's address for sending feedback
IPAddress pcIP;
uint16_t pcPort = 0;
bool pcConnected = false;
char packetBuffer[256]; // Buffer to hold incoming packets

// --- Stepper Setup ---
FastAccelStepperEngine engine = FastAccelStepperEngine();
FastAccelStepper *baseStepper = NULL;
FastAccelStepper *vertStepper = NULL;

#define BASE_STEP_PIN 42
#define BASE_DIR_PIN 41
#define VERT_STEP_PIN 36
#define VERT_DIR_PIN 35
#define RX_PIN 16
#define TX_PIN 17

const float baseStepsPerMM = 1600.0 / 95.0;
const float vertStepsPerMM = 3200.0 / 105.0;

const float BASE_MIN_MM = 0.0;
const float BASE_MAX_MM = 300.0;
const float VERT_MIN_MM = 0.0;
const float VERT_MAX_MM = 111.0;

const int NUM_SERVOS = 4;
u16 defaultSpeed[NUM_SERVOS];
u16 targetSpeed[NUM_SERVOS];
u8 defaultAcc[NUM_SERVOS];
s16 sendPos[NUM_SERVOS];
float p_values[6];
float v_values[6];
// --- Feetech Setup ---
SMS_STS feetech; 
u8 feetechIDs[4] = {1, 2, 3, 4}; 

// --- Feedback Timer ---
unsigned long lastFeedbackTime = 0;
const int feedbackIntervalMs = 100; // 10 Hz feedback

void setup() {
  Serial.begin(115200);
  Serial2.begin(1000000, SERIAL_8N1, 16, 17); 
  feetech.pSerial = &Serial2;

  // Connect to WiFi
  Serial.print("Connecting to WiFi");
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nConnected!");
  Serial.print("ESP32 IP Address: ");
  Serial.println(WiFi.localIP());

  // Start UDP
  udp.begin(localUdpPort);
  Serial.printf("Listening on UDP port %d\n", localUdpPort);

  for (int i = 0; i < NUM_SERVOS; i++) {
    defaultSpeed[i] = 3500;
    defaultAcc[i] = 100;
  }

  // Initialize FastAccelStepper Engine
  engine.init();
  
  // Configure Base Stepper
  baseStepper = engine.stepperConnectToPin(BASE_STEP_PIN);
  if (baseStepper) {
    baseStepper->setDirectionPin(BASE_DIR_PIN);
    baseStepper->setAcceleration(100000); 
    baseStepper->setCurrentPosition(0);
  }

  // Configure Vertical Stepper
  vertStepper = engine.stepperConnectToPin(VERT_STEP_PIN);
  if (vertStepper) {
    vertStepper->setDirectionPin(VERT_DIR_PIN);
    vertStepper->setAcceleration(100000); 
    vertStepper->setCurrentPosition(0);
  }
}

bool processCommand(const char* message, float* p_vals, float* v_vals) {
  int parsedItems = sscanf(
    message, 
    "P %f %f %f %f %f %f V %f %f %f %f %f %f END", 
    &p_vals[0], &p_vals[1], &p_vals[2], &p_vals[3], &p_vals[4], &p_vals[5],
    &v_vals[0], &v_vals[1], &v_vals[2], &v_vals[3], &v_vals[4], &v_vals[5]
  );

  // check if failed to parse
  if (parsedItems != 12) {
    Serial.println("Failed to parse command");
  }
  
  return (parsedItems == 12);
}

void loop() {
  // Handle Incoming UDP Packets
  int packetSize = udp.parsePacket();
  if (packetSize) {
    // Save PC's IP and port to route feedback back dynamically
    pcIP = udp.remoteIP();
    pcPort = udp.remotePort();
    pcConnected = true;

    // Read payload
    int len = udp.read(packetBuffer, 255);
    if (len > 0) {
      packetBuffer[len] = 0; // Null-terminate
    }
    
    if(processCommand(packetBuffer, p_values, v_values)){
      // --- Base rail stepper ---
      if (baseStepper) {
        float clampedBase = constrain(p_values[0], BASE_MIN_MM, BASE_MAX_MM);
        long targetSteps = (long)(clampedBase * baseStepsPerMM + 0.5);
        uint32_t speedSteps = (v_values[0] > 0)
          ? (uint32_t)(v_values[0] * baseStepsPerMM+0.5)
          : 10000; // default speed in steps/s
        baseStepper->setSpeedInHz(speedSteps);
        baseStepper->moveTo(targetSteps);
      }

      // --- Vertical rail stepper ---
      if (vertStepper) {
        float clampedVert = constrain(p_values[1], VERT_MIN_MM, VERT_MAX_MM);
        long targetSteps = (long)(clampedVert * vertStepsPerMM + 0.5);
        uint32_t speedSteps = (v_values[1] > 0)
          ? (uint32_t)(v_values[1] * vertStepsPerMM+0.5)
          : 10000; // default speed in steps/s
        vertStepper->setSpeedInHz(speedSteps);
        vertStepper->moveTo(targetSteps);
      }

      // --- Feetech servos ---
      for (int i = 0; i < 4; i++) {
        int servoID = feetechIDs[i];
        int sv = (int)(v_values[i + 2] + 0.5);

        if (sv == -1) {
          // Set current physical position as the middle (2048) via calibration
          feetech.CalibrationOfs(servoID);
        } else {
          sendPos[i] = (s16)(p_values[i + 2] + 0.5);
          if (sv != 0) {
            targetSpeed[i] = (u16)(abs(sv)+0.5);
          } else {
            targetSpeed[i] = defaultSpeed[i];
          }
        }
      }
      feetech.SyncWritePosEx(feetechIDs, NUM_SERVOS, sendPos, targetSpeed, defaultAcc); 
    }else{
      Serial.println("Bad command ignored");
    }
  }

  // Publish Feedback via UDP
  if (pcConnected && (millis() - lastFeedbackTime >= feedbackIntervalMs)) {
    lastFeedbackTime = millis();
    
    float basePosMM = baseStepper ? (baseStepper->getCurrentPosition() / baseStepsPerMM) : 0;
    float vertPosMM = vertStepper ? (vertStepper->getCurrentPosition() / vertStepsPerMM) : 0;
    
    // Construct feedback string
    String feedback = "FB," + String(basePosMM) + "," + String(vertPosMM);
    for (int i = 0; i < 4; i++) {
      int pos = feetech.ReadPos(feetechIDs[i]);
      feedback += "," + String(pos != -1 ? pos : 0);
    }
    feedback += "\n";
    // print feedback to serial port
//    Serial.print(feedback);

    // Send UDP packet
    udp.beginPacket(pcIP, pcPort);
    udp.print(feedback);
    udp.endPacket();
  }
}
