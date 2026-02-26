
import cv2
import time
import config

class CameraManager:
    def __init__(self):
        """
        Initializes the CameraManager object.

        Sets self.cap to None and then calls self.init_camera() to initialize the camera.

        :raises: Exception
        """
        self.cap = None
        self.init_camera()

    def init_camera(self):
        """Initialize camera with proper error handling"""
        try:
            self.cap = cv2.VideoCapture(config.CAMERA_IDX)
            if not self.cap.isOpened():
                raise Exception(f"Cannot open camera {config.CAMERA_IDX}")
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_W)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_H)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            ret, _ = self.cap.read()
            if not ret:
                raise Exception("Camera opened but cannot capture frames")
            print(f"✅ Camera initialized at {config.FRAME_W}x{config.FRAME_H}")
            return True
        except Exception as e:
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
            time.sleep(1)
            return self.init_camera()
        except Exception as e:
            print(f"❌ Camera reinitialization failed: {e}")
            return False

    def release(self):
        if self.cap is not None:
            self.cap.release()
            print("✅ Camera released")
