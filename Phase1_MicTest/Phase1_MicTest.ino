// ===== Phase 1: Mic / ADC sanity test (ESP32-C3 + MAX4466) =====
#include <Arduino.h>

#define PLOTTER_MODE 0            // 1 = Serial Plotter, 0 = readable text + diagnosis

constexpr uint8_t  MIC_PIN     = 1;       // GPIO1 = ADC1_CH1
constexpr uint32_t SAMPLE_RATE = 8000;
constexpr uint32_t PERIOD_US   = 1000000UL / SAMPLE_RATE;   // 125 us
constexpr uint32_t WINDOW_MS   = 50;
constexpr uint32_t WINDOW_N    = SAMPLE_RATE * WINDOW_MS / 1000;  // 400 samples

void setup() {
  Serial.begin(115200);
  delay(1500);

  analogReadResolution(12);                    // 0..4095
  analogSetPinAttenuation(MIC_PIN, ADC_11db);  // widest input range
  // NOTE: no pinMode() on purpose.

  for (int i = 0; i < 300; i++) {              // warm-up
    analogRead(MIC_PIN);
    delay(1);
  }
  Serial.println("Mic test started...");
}

void loop() {
  uint32_t sum = 0;
  uint16_t mn = 4095, mx = 0;

  uint32_t next = micros() + PERIOD_US;
  for (uint32_t i = 0; i < WINDOW_N; i++) {
    while ((int32_t)(micros() - next) < 0) { }
    next += PERIOD_US;
    uint16_t v = analogRead(MIC_PIN);
    sum += v;
    if (v < mn) mn = v;
    if (v > mx) mx = v;
  }

  uint16_t mean = sum / WINDOW_N;
  uint16_t p2p  = mx - mn;

#if PLOTTER_MODE
  Serial.printf("Min:%u,Max:%u,Mean:%u,P2P:%u\n", mn, mx, mean, p2p);
#else
  const char* status = "OK";
  if (mn <= 10 || mx >= 4090)         status = "CLIPPING -> lower the gain screw";
  else if (mean < 800 || mean > 3600) status = "BAD DC LEVEL -> check wiring/3.3V";
  else if (p2p <= 1)                  status = "FLAT LINE -> signal wire disconnected?";
  Serial.printf("min=%4u max=%4u mean=%4u p2p=%4u  [%s]\n", mn, mx, mean, p2p, status);
#endif
}