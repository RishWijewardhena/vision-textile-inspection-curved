# serial_communicator.py

import serial
import config
import random
import time


class SerialCommunicator:
    def __init__(self):
        """
        Initializes the SerialCommunicator object.

        Opens the serial port specified by config.SERIAL_PORT at the baud rate
        specified by config.BAUDRATE.
        """
        self.serial_port = None

        # Last known AI stitch length (mm). 0.0 means "not available yet".
        self.last_avg_stitch_length_mm = 0.0

        # Running total distance (mm)
        self.current_total_distance = 0.0

        # Anti-spam controls
        self._last_fallback_print_time = 0.0
        self._fallback_print_interval_sec = 2.0  # print fallback warning at most once per 2s

        try:
            self.serial_port = serial.Serial(config.SERIAL_PORT, config.BAUDRATE, timeout=0.1)
            print(f"[INFO] Opened serial port {config.SERIAL_PORT} at {config.BAUDRATE} baud")
        except Exception as e:
            print(f"[ERROR] Could not open serial port: {e}")
            self.serial_port = None

    @staticmethod
    def _fallback_stitch_length_mm() -> float:
        """Fallback stitch length when AI value is not yet available."""
        return round(random.uniform(6.0, 7.0), 3)

    def update_distance_from_stitch_count(self, data_line: str) -> bool:
        """Parse stitch count from Arduino and calculate total distance."""
        try:
            stitch_count = int(data_line.strip())

            avg_len = self.last_avg_stitch_length_mm
            used_fallback = False

            if avg_len is None or avg_len <= 0:
                avg_len = self._fallback_stitch_length_mm()
                used_fallback = True

                now = time.time()
                if now - self._last_fallback_print_time >= self._fallback_print_interval_sec:
                    print(
                        f"⚠️ Avg stitch length not available yet -> using fallback {avg_len:.3f}mm "
                        f"(Stitch count: {stitch_count})"
                    )
                    self._last_fallback_print_time = now

            self.current_total_distance = stitch_count * avg_len

            # Optional: reduce spam by printing only when not fallback or occasionally
            if not used_fallback:
                print(
                    f"📏 Updated total distance: {self.current_total_distance:.2f}mm "
                    f"(Stitches: {stitch_count}, Avg Length: {avg_len:.2f}mm)"
                )

            return True

        except ValueError:
            print(f"⚠️ Failed to parse stitch count: {data_line}")
            return False
        except Exception as e:
            print(f"⚠️ Error updating distance from stitch count: {e}")
            return False

    def read_serial_data(self):
        """Read data from the serial port and update the distance."""
        if not self.serial_port:
            return

        # IMPORTANT: keep buffer across calls, otherwise partial lines can be lost
        if not hasattr(self, "_buffer"):
            self._buffer = ""

        if self.serial_port.in_waiting:
            try:
                data = self.serial_port.read(self.serial_port.in_waiting).decode("utf-8", errors="ignore")
                self._buffer += data

                while "\n" in self._buffer:
                    line, self._buffer = self._buffer.split("\n", 1)
                    line = line.strip()
                    if line:
                        self.update_distance_from_stitch_count(line)

            except Exception as e:
                print(f"Warning: Serial read/decode error: {e}")
                self._buffer = ""

    def close(self):
        if self.serial_port is not None:
            try:
                self.serial_port.close()
            except Exception:
                pass
            print("✅ Serial port closed")