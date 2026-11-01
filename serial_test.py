import serial
import time

data = serial.Serial(
    'COM4',
    baudrate=9600,
    parity=serial.PARITY_NONE,
    stopbits=serial.STOPBITS_ONE,
    bytesize=serial.EIGHTBITS,
    timeout=1
)
import os 
def Read():
    print("Reading...")

    while True:
        Data = data.readline()
        Data = Data.decode('utf-8', 'ignore').strip()

        if Data == '':
            continue

        print("Received:", Data)

        
Read()
