const int analogPin = A0;

const float ironThreshold = 1.0;
const float nickelThreshold = 2.0;
const float cobaltThreshold = 3.0;

void setup() {
  Serial.begin(9600);
}

void loop() {
  int rawReading = analogRead(analogPin);
  float voltage = rawReading * (5.0 / 1023.0);

  if (voltage < ironThreshold) {
    Serial.println("Metal detected: Iron (Low concentration)");
  } else if (voltage >= ironThreshold && voltage < nickelThreshold) {
    Serial.println("Metal detected: Nickel");
  } else if (voltage >= nickelThreshold && voltage < cobaltThreshold) {
    Serial.println("Metal detected: Cobalt");
  } else {
    Serial.println("Metal detected: Unknown (High concentration)");
  }

  delay(100);
}

