# serial_communicator.py

import serial
import config
import time
from typing import Optional


class SerialCommunicator:
    def __init__(self,current_total_distance: float = 0.0):
        """
        Initializes the SerialCommunicator object.

        Opens the serial port specified by config.SERIAL_PORT at the baud rate
        specified by config.BAUDRATE.
        If no value is passed, it will automatically use 0.0.
        """
        self.serial_port = None

        # Last known AI stitch length (mm). 0.0 means "not available yet".
        self.last_avg_stitch_length_mm = 0.0
        self.current_total_distance = current_total_distance

        # Anti-spam controls
        self._last_fallback_print_time = 0.0
        self._fallback_print_interval_sec = 2.0  # print fallback warning at most once per 2s

        try:
            self.serial_port = serial.Serial(config.SERIAL_PORT, config.BAUDRATE, timeout=0.1)
            print(f"[INFO] Opened serial port {config.SERIAL_PORT} at {config.BAUDRATE} baud")
        except Exception as e:
            print(f"[ERROR] Could not open serial port: {e}")
            self.serial_port = None

    def update_distance_from_stitch_count(self, data_line: int) -> bool:
        """Update total distance using stitch delta (increment), not absolute count."""

        try:
            delta = int(data_line)
            avg_len = self.last_avg_stitch_length_mm

            if avg_len is None or avg_len <= 0:
                now = time.time()
                if now - self._last_fallback_print_time >= self._fallback_print_interval_sec:
                    print(
                        "⚠️ Avg stitch length not available yet; skipping distance update "
                        "(no fake data)."
                    )
                    self._last_fallback_print_time = now
                return False

            self.current_total_distance += delta * avg_len

            print(
                f"📏 Updated total distance: {self.current_total_distance:.2f}mm "
                f"(Delta: {delta}, Avg Length: {avg_len:.2f}mm)"
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
                        # self.update_distance_from_stitch_count(line)
                        last_stich_count = int(line)
                        return last_stich_count

            except Exception as e:
                print(f"Warning: Serial read/decode error: {e}")
                self._buffer = ""
        return None

    def close(self):
        if self.serial_port is not None:
            try:
                self.serial_port.close()
            except Exception:
                pass
            print("✅ Serial port closed")