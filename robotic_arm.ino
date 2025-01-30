#include <ESP32Servo.h>

#define SERVO_1_PIN 23
#define SERVO_2_PIN 22
#define SERVO_3_PIN 21
#define SERVO_4_PIN 19
#define SERVO_5_PIN 18
#define SERVO_6_PIN 5

Servo servo1;
Servo servo2;
Servo servo3;
Servo servo4;
Servo servo5;
Servo servo6;

const float L1 = 10.0;
const float L2 = 15.0;
const float L3 = 15.0;

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
  float theta4 = wristAngle;
  float theta5 = 90;
  
  setServoAngles(theta1, theta2, theta3, theta4, theta5, clawAngle);
}

void setup() {
  servo1.attach(SERVO_1_PIN);
  servo2.attach(SERVO_2_PIN);
  servo3.attach(SERVO_3_PIN);
  servo4.attach(SERVO_4_PIN);
  servo5.attach(SERVO_5_PIN);
  servo6.attach(SERVO_6_PIN);

  setServoAngles(90, 90, 90, 90, 90, 90);
}

void loop() {
  float targetX = 10.0;
  float targetY = 10.0;
  float targetZ = 10.0;
  float wristAngle = 45.0;
  float clawAngle = 30.0;

  inverseKinematics(targetX, targetY, targetZ, wristAngle, clawAngle);

  delay(1000);
}

