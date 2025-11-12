#include <Arduino.h>

constexpr int PIR_PIN = 14;     // input from PIR
constexpr int PI_OUT  = 27;     // goes to Pi GPIO17 via 1k series resistor

// tweak these
constexpr uint32_t PULSE_MS       = 150;   // pulse to Pi on motion
constexpr uint32_t RETRIGGER_MS   = 500;   // ignore re-triggers for this long
constexpr uint32_t STABLE_MS      = 30;    // basic deglitch

unsigned long lastChange = 0;
int lastStable = LOW;

void setup() {
  pinMode(PIR_PIN, INPUT);           // most PIRs have their own pull; add INPUT_PULLDOWN if yours floats low
  pinMode(PI_OUT, OUTPUT);
  digitalWrite(PI_OUT, LOW);
  Serial.begin(115200);
}

void loop() {
  int raw = digitalRead(PIR_PIN);
  unsigned long now = millis();

  // simple stability check
  static int prev = raw;
  static unsigned long tEdge = now;
  if (raw != prev) { tEdge = now; prev = raw; }

  if ((now - tEdge) >= STABLE_MS && raw != lastStable) {
    lastStable = raw;
    lastChange = now;

    if (lastStable == HIGH) {
      // motion detected: send a clean pulse to the Pi
      digitalWrite(PI_OUT, HIGH);
      delay(PULSE_MS);
      digitalWrite(PI_OUT, LOW);
      Serial.println("Motion -> pulse sent to Pi");
    }
  }

  // optional: lockout window to avoid spamming (PIRs often hold HIGH for seconds anyway)
  if (lastStable == HIGH && (now - lastChange) < RETRIGGER_MS) {
    // nothing
  }
}