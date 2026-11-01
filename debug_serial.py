# debug_serial.py
import serial, time
ser = serial.Serial("COM9", 9600, timeout=1)
print("Port open. Waiting for bytes...")
t0 = time.time()
while time.time() - t0 < 10:
    raw = ser.readline()
    if raw:
        print(f"RAW: {raw!r}")
    else:
        print("  (no data in 1s)")
ser.close()