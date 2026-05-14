#include <PDM.h>

const int SAMPLE_RATE = 16000;
const int BUFFER_SIZE = 512;

short sampleBuffer[BUFFER_SIZE];
volatile int samplesRead = 0;

void onPDMdata() {
  int bytesAvailable = PDM.available();

  // Read into the sample buffer
  PDM.read(sampleBuffer, bytesAvailable);

  // Convert bytes to sample count
  samplesRead = bytesAvailable / 2;
}

void setup() {
  Serial.begin(921600);

  while (!Serial);

  // Configure the microphone
  PDM.onReceive(onPDMdata);

  // 1 channel = mono
  if (!PDM.begin(1, SAMPLE_RATE)) {
    Serial.println("Failed to start PDM!");
    while (1);
  }

  // Optional gain adjustment
  PDM.setGain(30);

  delay(1000);
}

void loop() {
  if (samplesRead) {

    // Send raw binary PCM samples over serial
    Serial.write((uint8_t*)sampleBuffer, samplesRead * 2);

    samplesRead = 0;
  }
}