import serial
import time

PORT = 'COM7'
BAUD = 115200

def get_serial_connection():
    # Serial Connection object
    ser = serial.Serial(PORT, BAUD, timeout=1)
    time.sleep(2)  # allow Arduino to reset
    return ser


def triggerBuzzer():
    with get_serial_connection() as esp:
        esp.write(b'W') # Send 'W' serially
        time.sleep(1)