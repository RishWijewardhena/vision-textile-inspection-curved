# serial_communicator.py

import serial
import config
import time
import threading
from typing import Optional
from utils.resource_discovery import find_esp32


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
        self._last_reconnect_attempt = 0.0
        self._reconnect_interval_sec = 2.0
        self._serial_lock = threading.Lock()

        self._open_serial_port()

    def _open_serial_port(self) -> bool:
        """Open configured serial port, then fall back to discovered ESP32 port."""
        discovered_port = find_esp32()
        candidate_ports = [config.SERIAL_PORT]
        if discovered_port and discovered_port not in candidate_ports:
            candidate_ports.append(discovered_port)

        for port in candidate_ports:
            try:
                self.serial_port = serial.Serial(port, config.BAUDRATE, timeout=0.1)
                print(f"[INFO] Opened serial port {port} at {config.BAUDRATE} baud")
                return True
            except Exception as exc:
                print(f"[WARN] Could not open serial port {port}: {exc}")

        self.serial_port = None
        print("[ERROR] Serial port unavailable after configured + auto-discovery attempts")
        return False

    def _try_reconnect(self):
        """Attempt reconnect with rate limiting to avoid busy-loop retries."""
        now = time.time()
        if now - self._last_reconnect_attempt < self._reconnect_interval_sec:
            return
        self._last_reconnect_attempt = now
        print("[INFO] Serial port not available, trying reconnect...")
        self._open_serial_port()

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
            self._try_reconnect()
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
                        try:
                            last_stich_count = int(line)
                            return last_stich_count
                        except ValueError:
                            print(f"[WARN] Non-integer serial line ignored: {line}")
                            continue

            except Exception as e:
                print(f"Warning: Serial read/decode error: {e}")
                try:
                    self.serial_port.close()
                except Exception:
                    pass
                self.serial_port = None
                self._buffer = ""
                self._try_reconnect()
        return None

    def send_command(self, command) -> bool:
        """Send a command to ESP32 and return True on success."""
        if not isinstance(command, str):
            command = str(command)

        if not command:
            return False

        if not self.serial_port or not self.serial_port.is_open:
            self._try_reconnect()

        if not self.serial_port or not self.serial_port.is_open:
            print("⚠️ Cannot send serial command: no active serial connection")
            return False

        try:
            with self._serial_lock:
                self.serial_port.write(command.encode("utf-8"))
                self.serial_port.flush()

            if getattr(config, "LOG_DEBUG", False):
                print(f"📤 Serial command sent: {command}")
            return True
        except Exception as e:
            print(f"❌ Failed to send serial command '{command}': {e}")
            try:
                self.serial_port.close()
            except Exception:
                pass
            self.serial_port = None
            self._try_reconnect()
            return False

    def close(self):
        if self.serial_port is not None:
            try:
                self.serial_port.close()
            except Exception:
                pass
            print("✅ Serial port closed")

if __name__ == "__main__":
    communicator = SerialCommunicator()
    try:
        while True:
            stitch_count = communicator.read_serial_data()
            if stitch_count is not None:
                print(f"Received stitch count: {stitch_count}")
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        communicator.close()