import cv2
import os
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


def _normalize_camera_index(value):
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit():
            return int(stripped)
        return stripped
    return value


def list_camera_candidates(preferred=None, cam_list=None, include_by_id=True):
    """Return an ordered, de-duplicated list of camera candidates."""
    candidates = []

    preferred = _normalize_camera_index(preferred)
    if preferred is not None:
        candidates.append(preferred)

    if include_by_id and os.path.isdir("/dev/v4l/by-id"):
        for name in sorted(os.listdir("/dev/v4l/by-id")):
            candidates.append(os.path.join("/dev/v4l/by-id", name))

    candidates.extend(cam_list or DEFAULT_CAM_LIST)

    deduped = []
    seen = set()
    for candidate in candidates:
        key = (type(candidate).__name__, str(candidate))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)

    return deduped


def find_camera(cam_list=None, preferred=None, include_by_id=True):
    """Return first camera path that can be opened, else first candidate."""
    candidates = list_camera_candidates(
        preferred=preferred,
        cam_list=cam_list,
        include_by_id=include_by_id,
    )

    for cam in candidates:
        cap = cv2.VideoCapture(cam)
        if cap.isOpened():
            cap.release()
            return cam
        cap.release()

    return candidates[0] if candidates else "/dev/video0"
