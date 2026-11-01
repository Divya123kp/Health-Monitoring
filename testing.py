import serial

PORT = "COM4"
BAUD = 9600

data = serial.Serial(
    PORT,
    baudrate=BAUD,
    parity=serial.PARITY_NONE,
    stopbits=serial.STOPBITS_ONE,
    bytesize=serial.EIGHTBITS,
    timeout=1,
)

print(f"Listening on {PORT} @ {BAUD}...")
while True:
    raw = data.readline()
    if not raw:
        continue
    line = raw.decode("utf-8", "ignore").strip()

    parts = line.split(",")
    if len(parts) != 4:
        print(f"[skip] malformed: {line!r}")
        continue

    try:
        IR, RED, HR, SpO2 = (float(p) for p in parts)
    except ValueError:
        print(f"[skip] non-numeric: {line!r}")
        continue

    print(f"IR={IR:.0f}  RED={RED:.0f}  HR={HR:.0f}  SpO2={SpO2:.1f}")