#include <ESP32Servo.h>
#include <esp_now.h>
#include <WiFi.h>
#include <cmath>

#define SERVO_1_PIN 23
#define SERVO_2_PIN 22
#define SERVO_3_PIN 21
#define SERVO_4_PIN 19
#define SERVO_5_PIN 18
#define SERVO_6_PIN 5

Servo servo1, servo2, servo3, servo4, servo5, servo6;
const float L1 = 10.0, L2 = 15.0, L3 = 15.0;

struct SensorData {
    float moisture;
    float temperature;
    float humidity;
    bool objectDetected;
    float objectX;
    float objectY;
    float objectZ;
};

SensorData incomingData;
float movementPattern[4][3] = {{10, 10, 5}, {15, 5, 10}, {5, 15, 5}, {10, 5, 15}};
int patternIndex = 0;

void OnDataRecv(const uint8_t *mac, const uint8_t *incomingDataRaw, int len) {
    memcpy(&incomingData, incomingDataRaw, sizeof(incomingData));
    Serial.printf("Moisture: %.2f, Temperature: %.2f, Humidity: %.2f, Object Detected: %s, Object Coordinates: (%.2f, %.2f, %.2f)\n",
                  incomingData.moisture, incomingData.temperature,
                  incomingData.humidity, incomingData.objectDetected ? "Yes" : "No",
                  incomingData.objectX, incomingData.objectY, incomingData.objectZ);
}

void setServoAngles(float theta1, float theta2, float theta3, float theta4, float theta5, float theta6) {
    servo1.write(constrain(theta1, 0, 180));
    servo2.write(constrain(theta2, 0, 180));
    servo3.write(constrain(theta3, 0, 180));
    servo4.write(constrain(theta4, 0, 180));
    servo5.write(constrain(theta5, 0, 180));
    servo6.write(constrain(theta6, 0, 180));
}

void inverseKinematics(float x, float y, float z, float wristAngle, float clawAngle) {
    float theta1 = atan2(y, x) * 180 / PI;
    float r = sqrt(x * x + y * y);
    float d = sqrt(r * r + z * z);
    float alpha = atan2(z, r);
    float beta = acos((L2 * L2 + d * d - L3 * L3) / (2 * L2 * d));
    float gamma = acos((L2 * L2 + L3 * L3 - d * d) / (2 * L2 * L3));
    float theta2 = (alpha + beta) * 180 / PI;
    float theta3 = gamma * 180 / PI;
    setServoAngles(theta1, theta2, theta3, wristAngle, 90, clawAngle);
}

void setup() {
    Serial.begin(115200);
    servo1.attach(SERVO_1_PIN);
    servo2.attach(SERVO_2_PIN);
    servo3.attach(SERVO_3_PIN);
    servo4.attach(SERVO_4_PIN);
    servo5.attach(SERVO_5_PIN);
    servo6.attach(SERVO_6_PIN);
    setServoAngles(90, 90, 90, 90, 90, 90);
    WiFi.mode(WIFI_STA);
    if (esp_now_init() == ESP_OK) {
        esp_now_register_recv_cb(OnDataRecv);
    }
}

void loop() {
    if (incomingData.objectDetected) {
        Serial.println("Target object detected by ESP32-CAM. Initiating precision pick-up sequence.");
        inverseKinematics(incomingData.objectX, incomingData.objectY, incomingData.objectZ, 45.0, 30.0);
        delay(1000);
        setServoAngles(90, 90, 90, 90, 90, 90);
    } else if (incomingData.moisture > 50) {
        Serial.println("High moisture level detected. Adjusting grip and picking up wet object.");
        inverseKinematics(5.0, 5.0, 5.0, 60.0, 40.0);
        delay(1000);
        setServoAngles(90, 90, 90, 90, 90, 90);
    } else if (incomingData.temperature > 30) {
        Serial.println("High temperature detected. Using special handling technique.");
        inverseKinematics(15.0, 5.0, 5.0, 30.0, 20.0);
        delay(1000);
        setServoAngles(90, 90, 90, 90, 90, 90);
    } else {
        Serial.println("No immediate task detected. Performing scanning movement.");
        float *pos = movementPattern[patternIndex];
        inverseKinematics(pos[0], pos[1], pos[2], 45.0, 30.0);
        patternIndex = (patternIndex + 1) % 4;
        delay(1500);
    }
    delay(500);
}

