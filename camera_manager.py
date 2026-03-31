
import cv2
import time
import config
from utils.resource_discovery import find_camera

class CameraManager:
    def __init__(self):
        """
        Initializes the CameraManager object.

        Sets self.cap to None and then calls self.init_camera() to initialize the camera.

        :raises: Exception
        """
        self.cap = None
        self.camera_idx = config.CAMERA_IDX
        self.init_camera()

    def init_camera(self):
        """Initialize camera with proper error handling"""
        preferred_cam = self.camera_idx
        discovered_cam = find_camera()
        candidates = [preferred_cam]
        if discovered_cam and discovered_cam not in candidates:
            candidates.append(discovered_cam)

        last_error = None

        for cam_idx in candidates:
            try:
                cap = cv2.VideoCapture(cam_idx)
                if not cap.isOpened():
                    raise Exception(f"Cannot open camera {cam_idx}")

                cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_W)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_H)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

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
                print(f"⚠️ Camera init attempt failed on {cam_idx}: {e}")

        try:
            self.cap = None
            print(f"❌ Camera initialization failed: {last_error}")
            return False
        except Exception as e:
            self.cap = None
            print(f"❌ Camera initialization failed: {e}")
            return False

    def capture_frame_safely(self):
        """Safely capture a frame with error handling and buffer flushing"""
        try:
            for _ in range(3):
                ret, _ = self.cap.read()
                if not ret:
                    break
            ret, frame = self.cap.read()
            if not ret:
                print("❌ ERROR: Failed to capture frame")
                if self.reinit_camera():
                    ret, frame = self.cap.read()
                    if ret:
                        print("✅ Camera reinitialized successfully")
                        return frame
                return None
            return frame
        except Exception as e:
            print(f"❌ Camera capture error: {e}")
            if self.reinit_camera():
                try:
                    ret, frame = self.cap.read()
                    if ret:
                        return frame
                except:
                    pass
            return None

    def reinit_camera(self):
        """Attempt to reinitialize camera if it becomes unavailable"""
        try:
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            time.sleep(1)
            return self.init_camera()
        except Exception as e:
            print(f"❌ Camera reinitialization failed: {e}")
            return False

    def release(self):
        if self.cap is not None:
            self.cap.release()
            print("✅ Camera released")
