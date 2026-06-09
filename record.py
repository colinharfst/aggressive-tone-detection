import serial
import wave
import time

PORT = 'COM3'
BAUD = 921600

OUTPUT_FILE = 'recording.wav'

SAMPLE_RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH = 2   # 16-bit audio

RECORD_SECONDS = 6

ser = serial.Serial(PORT, BAUD)

print("Recording...")

frames = bytearray()

start = time.time()

while time.time() - start < RECORD_SECONDS:
    data = ser.read(1024)
    frames.extend(data)

print("Done.")

ser.close()

with wave.open(OUTPUT_FILE, 'wb') as wf:
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(SAMPLE_WIDTH)
    wf.setframerate(SAMPLE_RATE)
    wf.writeframes(frames)

print(f"Saved to {OUTPUT_FILE}")