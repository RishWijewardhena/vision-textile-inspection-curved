import cv2
import os
import time
import config
from utils.resource_discovery import list_camera_candidates

class CameraManager:
    def __init__(self, reload_callback=None):
        """
        Initializes the CameraManager object.

        Sets self.cap to None and then calls self.init_camera() to initialize the camera.

        :raises: Exception
        """
        self.cap = None
        self.camera_idx = None
        self.reconnect_attempts = 0
        self.reload_callback = reload_callback
        self.init_camera()

    def init_camera(self):
        """Initialize camera with proper error handling"""
        candidates = list_camera_candidates(preferred=config.CAMERA_IDX)
        if candidates:
            print(f"[INFO] Camera candidates: {', '.join(str(c) for c in candidates)}")
        else:
            print("[WARN] No camera candidates discovered")

        last_error = None

        for cam_idx in candidates:
            try:
                if isinstance(cam_idx, str) and cam_idx.startswith("/dev/"):
                    if not os.path.exists(cam_idx):
                        raise FileNotFoundError(f"Device path not found: {cam_idx}")
                    if not os.access(cam_idx, os.R_OK | os.W_OK):
                        raise PermissionError(
                            f"No access to {cam_idx} (add user to video group)"
                        )

                cap = cv2.VideoCapture(cam_idx,cv2.CAP_V4L2)
                if not cap.isOpened():
                    raise Exception(f"Cannot open camera {cam_idx}")

                # Force MJPG compression FIRST (before resolution settings for proper negotiation)
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_W)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_H)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                time.sleep(1)

                ret, _ = cap.read()
                if not ret:
                    cap.release()
                    raise Exception(f"Camera opened but cannot capture frames ({cam_idx})")

                self.cap = cap
                self.camera_idx = cam_idx
                print(f"✅ Camera initialized on {cam_idx} at {config.FRAME_W}x{config.FRAME_H}")
                return True
            except Exception as e:
                last_error = e
                print(f" Camera init attempt failed on {cam_idx}: {e}")

        try:
            self.cap = None
            print(f" Camera initialization failed: {last_error}")
            return False
        except Exception as e:
            self.cap = None
            print(f" Camera initialization failed: {e}")
            return False

    def capture_frame_safely(self):
        """Safely capture a frame with error handling and buffer flushing"""
        try:
            if self.cap is None:
                print(" Camera not initialized; attempting to reinitialize")
                if not self.reinit_camera():
                    self._handle_reconnect_failure()
                    return None

            # Drain a few buffered frames to reduce stale captures without adding long latency.
            flush_frames = max(0, int(getattr(config, "CAMERA_FLUSH_FRAMES", 2)))
            for _ in range(flush_frames):
                if not self.cap.grab():
                    break

            ret, frame = self.cap.read()
            if not ret:
                print(" ERROR: Failed to capture frame")
                self._handle_reconnect_failure()
                return None

            self.reconnect_attempts = 0
            return frame
        except Exception as e:
            print(f" Camera capture error: {e}")
            self._handle_reconnect_failure()
            return None

    def _handle_reconnect_failure(self):
        self.reconnect_attempts += 1
        print(
            f" Camera reconnect attempt {self.reconnect_attempts}/"
            f"{config.MAX_RECONNECT_ATTEMPTS}"
        )

        if self.reconnect_attempts >= config.MAX_RECONNECT_ATTEMPTS:
            print(" Camera disconnected. Reloading webcam driver and attempting reconnect...")
            if self.reload_callback:
                try:
                    self.reload_callback()
                except Exception as exc:
                    print(f" Camera reload callback failed: {exc}")

            self.reinit_camera()
            self.reconnect_attempts = 0
        else:
            self.reinit_camera()

    def reinit_camera(self):
        """Attempt to reinitialize camera if it becomes unavailable"""
        try:
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            time.sleep(1)
            
            return self.init_camera()
        except Exception as e:
            print(f" Camera reinitialization failed: {e}")
            return False

    def release(self):
        if self.cap is not None:
            self.cap.release()
            print("✅ Camera released")
