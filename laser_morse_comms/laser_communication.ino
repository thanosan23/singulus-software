
int LASER_PIN = 2;
int PHOTO_PIN = A1;

const int sensorThreshold = 2.3;

const unsigned long DOT_DURATION = 500; 
const unsigned long DASH_DURATION = 1500;
const unsigned long SYMBOL_GAP = 700;
const unsigned long LETTER_GAP = 2000;
const unsigned long WORD_GAP = 4000;


struct MorseMapping {
  char letter;
  const char* code;
};

MorseMapping morseTable[] = {
  {'A', ".-"},
  {'B', "-..."},
  {'C', "-.-."},
  {'D', "-.."},
  {'E', "."},
  {'F', "..-."},
  {'G', "--."},
  {'H', "...."},
  {'I', ".."},
  {'J', ".---"},
  {'K', "-.-"},
  {'L', ".-.."},
  {'M', "--"},
  {'N', "-."},
  {'O', "---"},
  {'P', ".--."},
  {'Q', "--.-"},
  {'R', ".-."},
  {'S', "..."},
  {'T', "-"},
  {'U', "..-"},
  {'V', "...-"},
  {'W', ".--"},
  {'X', "-..-"},
  {'Y', "-.--"},
  {'Z', "--.."},
  {'1', ".----"},
  {'2', "..---"},
  {'3', "...--"},
  {'4', "....-"},
  {'5', "....."},
  {'6', "-...."},
  {'7', "--..."},
  {'8', "---.."},
  {'9', "----."},
  {'0', "-----"}
};
const int morseTableSize = sizeof(morseTable) / sizeof(morseTable[0]);

const char* getMorseForChar(char c) {
  for (int i = 0; i < morseTableSize; i++) {
    if (morseTable[i].letter == c) {
      return morseTable[i].code;
    }
  }
  return "";
}

char decodeMorse(String morse) {
  for (int i = 0; i < morseTableSize; i++) {
    if (morse.equals(morseTable[i].code)) {
      return morseTable[i].letter;
    }
  }
  return '?';
}

// Transmitter Machine

enum MsgState {idle, laser_on, laser_off, letter_gap};
MsgState msgState = idle;
String msgMessage = ""; // The message to be transmitted
int msgMessageIndex = 0; // Current character in the message
String msgMorse = ""; // Morse code for the current character
int msgMorseIndex = 0; // Current character in the Morse
bool msgActive = false; // Transmission is in progress (True/False)
unsigned long msgStateStartTime = 0; // uses millis()
unsigned long msgDuration = 0; // Duration for the current state

void processTransmitter() {
  if (!msgActive) return;
  
  unsigned long now = millis();
  
  switch(msgState) {
    case idle:
      if (msgMessageIndex < msgMessage.length()) {
        char c = msgMessage.charAt(msgMessageIndex);
        if (c == ' ') {
          msgState = letter_gap;
          msgDuration = WORD_GAP;
          msgStateStartTime = now;
          msgMessageIndex++;
        } else {
          msgMorse = String(getMorseForChar(c));
          msgMorseIndex = 0;
          if (msgMorse.length() == 0) {
            msgMessageIndex++;
          } else {
            msgState = laser_on;
            msgStateStartTime = now;
          }
        }
      }
      else {
        msgActive = false;
        Serial.println("  [Transmission complete]");
      }
      break;
      
    case laser_on: {
        if (msgMorseIndex < msgMorse.length()) {
          char symbol = msgMorse.charAt(msgMorseIndex);
          digitalWrite(LASER_PIN, HIGH);
          if (symbol == '.')
            msgDuration = DOT_DURATION;
          else if (symbol == '-')
            msgDuration = DASH_DURATION;
          else
            msgDuration = DOT_DURATION;

          if (now - msgStateStartTime >= msgDuration) {
            digitalWrite(LASER_PIN, LOW);
            msgState = laser_off;
            msgStateStartTime = now;
            msgDuration = SYMBOL_GAP;
          }
        } else {
          msgState = letter_gap;
          msgStateStartTime = now;
          msgDuration = LETTER_GAP;
        }
      }
      break;
      
    case laser_off:
      if (now - msgStateStartTime >= msgDuration) {
        msgMorseIndex++;
        if (msgMorseIndex < msgMorse.length()) {
          msgState = laser_on;
          msgStateStartTime = now;
        } else {
          msgState = letter_gap;
          msgStateStartTime = now;
          msgDuration = LETTER_GAP;
          msgMessageIndex++;
        }
      }
      break;
      
    case letter_gap:
      if (now - msgStateStartTime >= msgDuration) {
        msgState = idle;
      }
      break;
  }
}

// Reciever Machine

bool lastLightState = false; // Current state: true=light, false=dark
unsigned long lastStateChangeTime = 0;
String rxMorseLetter = ""; // Morse for the current letter
String rxMessage = ""; // Received message so far

void processReceiver() {
  unsigned long now = millis();
  float sensorVal = (analogRead(PHOTO_PIN))/1023 * 5.0;
  bool currentLightState = (sensorVal > sensorThreshold);
  
  if (currentLightState != lastLightState) {
    unsigned long duration = now - lastStateChangeTime;
    lastStateChangeTime = now;
    
    if (lastLightState == true && !currentLightState) {
      if (duration >= 300 && duration < 700)
        rxMorseLetter += ".";
      else if (duration >= 700 && duration < 1300)
        rxMorseLetter += "-";
    }
    else if (lastLightState == false && currentLightState) {
      if (duration >= (LETTER_GAP - 500) && duration < WORD_GAP) {
        if (rxMorseLetter.length() > 0) {
          char decoded = decodeMorse(rxMorseLetter);
          rxMessage += decoded;
          Serial.print(decoded);
          rxMorseLetter = "";
        }
      }
      else if (duration >= WORD_GAP) {
        if (rxMorseLetter.length() > 0) {
          char decoded = decodeMorse(rxMorseLetter);
          rxMessage += decoded;
          Serial.print(decoded);
          rxMorseLetter = "";
        }
        rxMessage += " ";
        Serial.print(" ");
      }
    }
    
    lastLightState = currentLightState;
  }
  
  if (!currentLightState && (now - lastStateChangeTime > (LETTER_GAP + 500)) && rxMorseLetter.length() > 0) {
    char decoded = decodeMorse(rxMorseLetter);
    rxMessage += decoded;
    Serial.print(decoded);
    rxMorseLetter = "";
  }
}

void setup() {
  Serial.begin(9600);
  Serial.println("Arduino Morse Code Communication System");
  
  pinMode(LASER_PIN, OUTPUT);
  digitalWrite(LASER_PIN, LOW);
  
  pinMode(PHOTO_PIN, INPUT);
  lastStateChangeTime = millis();
}

void loop() {
  if (Serial.available() > 0) {
    String input = Serial.readStringUntil('\n');
    input.trim();
    if (input.length() > 0) {
      msgMessage = input;
      msgMessage.toUpperCase();
      msgMessageIndex = 0;
      msgActive = true;
      msgState = idle;
      Serial.print("Transmitting: ");
      Serial.println(msgMessage);
    }
  }
  
  processTransmitter();
  processReceiver();
}


