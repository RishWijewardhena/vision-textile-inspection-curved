
import serial
import sys
import config

class SerialCommunicator:
    def __init__(self):
        """
        Initializes the SerialCommunicator object.

        Tries to open the serial port specified by config.SERIAL_PORT at
        the baud rate specified by config.BAUDRATE. If successful, prints
        an informational message. If unsuccessful, prints an error
        message and sets self.serial_port to None.

        :raises: Exception
        """
        self.serial_port = None
        self.last_avg_stitch_length_mm = 0.0
        self.current_total_distance = 0.0
        try:
            self.serial_port = serial.Serial(config.SERIAL_PORT, config.BAUDRATE, timeout=0.1)
            print(f"[INFO] Opened serial port {config.SERIAL_PORT} at {config.BAUDRATE} baud")
        except Exception as e:
            print(f"[ERROR] Could not open serial port: {e}")
            self.serial_port = None

    def update_distance_from_stitch_count(self, data_line):
        """Parse stitch count from Arduino and calculate total distance."""
        try:
            stitch_count = int(data_line.strip())
            if self.last_avg_stitch_length_mm > 0:
                self.current_total_distance = stitch_count * self.last_avg_stitch_length_mm
                print(f"📏 Updated total distance: {self.current_total_distance:.2f}mm (Stitches: {stitch_count}, Avg Length: {self.last_avg_stitch_length_mm:.2f}mm)")
            else:
                print(f"⚠️ Cannot calculate distance: Average stitch length is not available yet. (Stitch count: {stitch_count})")
            return True
        except ValueError:
            print(f"⚠️ Failed to parse stitch count: {data_line}")
            return False

    def read_serial_data(self):
        """Read data from the serial port and update the distance."""
        buffer = ""
        if self.serial_port.in_waiting:
            try:
                data = self.serial_port.read(self.serial_port.in_waiting).decode('utf-8', errors='ignore')
                buffer += data
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    line = line.strip()
                    if line:
                        self.update_distance_from_stitch_count(line)
            except UnicodeDecodeError:
                print("Warning: Invalid UTF-8 data from Arduino")
                buffer = ""

    def close(self):
        if self.serial_port is not None:
            self.serial_port.close()
            print("✅ Serial port closed")
