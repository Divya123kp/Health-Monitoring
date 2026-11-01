"""
serial_reader.py
----------------
Reads lines from a serial port in the format produced by the ESP/Arduino:

    96054,68839,93,84\r\n

=> IR, RED, HR, SpO2

Pushes a dict {IR, RED, HR, SpO2} into a queue consumed by app.py.
Handles partial reads, boot banners, and CR/LF variants.
"""
import threading
import time
from queue import Queue

try:
    import serial
except ImportError:
    serial = None


class SerialReader(threading.Thread):
    def __init__(self, port="COM4", baud=9600, out_queue: Queue = None):
        super().__init__(daemon=True)
        self.port = port
        self.baud = baud
        self.queue = out_queue if out_queue is not None else Queue()
        self._stop_evt = threading.Event()  # NOT "_stop": that shadows Thread._stop()

    # -- parsing -----------------------------------------------------------
    @staticmethod
    def parse_line(line: str):
        """
        '96054,68839,93,84\r\n' -> {'IR':96054.0,'RED':68839.0,'HR':93.0,'SpO2':84.0}
        Returns None for malformed lines.
        """
        line = line.strip()
        if not line:
            return None
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 4:
            return None
        try:
            IR, RED, HR, SpO2 = (float(p) for p in parts)
        except ValueError:
            return None
        return {"IR": IR, "RED": RED, "HR": HR, "SpO2": SpO2}

    # -- thread ------------------------------------------------------------
    def run(self):
        if serial is None:
            print("[serial] pyserial not installed — reader disabled")
            return

        while not self._stop_evt.is_set():
            try:
                with serial.Serial(
                    self.port,
                    baudrate=self.baud,
                    parity=serial.PARITY_NONE,
                    stopbits=serial.STOPBITS_ONE,
                    bytesize=serial.EIGHTBITS,
                    timeout=1,
                ) as ser:
                    print(f"[serial] opened {self.port} @ {self.baud}")
                    while not self._stop_evt.is_set():
                        raw = ser.readline()
                        if not raw:
                            continue
                        line = raw.decode("utf-8", "ignore")
                        reading = self.parse_line(line)
                        if reading:
                            self.queue.put(reading)
                        else:
                            print(f"[serial] skipped line: {line.strip()!r}")
            except Exception as e:
                print(f"[serial] error: {e} — retrying in 2 s")
                time.sleep(2)

    def stop(self):
        self._stop_evt.set()