import cv2
import serial.tools.list_ports


ESP32_VID = 0x303A
ESP32_PID = 0x1001
DEFAULT_CAM_LIST = [
    "/dev/video0",
    "/dev/video1",
    "/dev/video2",
    "/dev/video3",
    "/dev/video4",
]


def list_serial_ports():
    """Return a list of detected serial ports with metadata for diagnostics."""
    ports = []
    try:
        for port in serial.tools.list_ports.comports():
            ports.append(
                {
                    "device": port.device,
                    "vid": port.vid,
                    "pid": port.pid,
                    "hwid": port.hwid,
                    "manufacturer": port.manufacturer,
                    "product": port.product,
                    "description": port.description,
                }
            )
    except Exception as exc:
        print(f"[WARN] Serial port listing failed: {exc}")
    return ports


def find_esp32():
    """Return serial device path for ESP32 by USB VID/PID, or None."""
    try:
        for port in serial.tools.list_ports.comports():
            if port.vid == ESP32_VID and port.pid == ESP32_PID:
                return port.device
    except Exception as exc:
        print(f"[WARN] Serial port discovery failed: {exc}")
    return None


def find_camera(cam_list=None):
    """Return first camera path that can be opened, else first candidate."""
    candidates = cam_list or DEFAULT_CAM_LIST

    for cam in candidates:
        cap = cv2.VideoCapture(cam)
        if cap.isOpened():
            cap.release()
            return cam
        cap.release()

    return candidates[0] if candidates else "/dev/video0"
