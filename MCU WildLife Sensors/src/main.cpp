#include <Arduino.h>

const int PIR_PIN = 13;   // change if your PIR uses another GPIO
const int LED_PIN = 4;    // onboard LED on many ESP32 dev boards

void setup() {
  Serial.begin(115200);
  pinMode(PIR_PIN, INPUT);
  pinMode(LED_PIN, OUTPUT);
  Serial.println("Motion sensor test starting...");
}

void loop() {
  int motion = digitalRead(PIR_PIN);
  digitalWrite(LED_PIN, motion ? HIGH : LOW);
  if (motion) Serial.println("Motion detected!");
  delay(200);
}