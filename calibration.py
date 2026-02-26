
import json
import cv2
import numpy as np
import config

def load_json(path):
    """Load and parse a JSON file."""
    with open(path, 'r') as f:
        return json.load(f)

def compute_camera_plane(R, t):
    """Compute fabric plane in camera coordinates from extrinsics."""
    n_c = R[:, 2].astype(np.float64)
    d_c = -float(n_c.dot(t))
    return n_c, d_c

def pixel_to_world_using_camera_plane(u, v, K, dist, R, t, n_c, d_c):
    """Convert 2D pixel to 3D world coordinate via ray-plane intersection."""
    try:
        pts = np.array([[[float(u), float(v)]]], dtype=np.float64)
        und = cv2.undistortPoints(pts, K, dist, P=None)
        x_n, y_n = float(und[0,0,0]), float(und[0,0,1])
        d_cam = np.array([x_n, y_n, 1.0], dtype=np.float64)
        denom = float(n_c.dot(d_cam))
        if abs(denom) < 1e-9:
            return None
        s = -d_c / denom
        X_cam = s * d_cam
        X_world = R.T.dot(X_cam - t)
        return X_world
    except Exception:
        return None

def get_mm_per_pixel():
    """
    Calculates the mm per pixel conversion factor.
    It first tries to calculate it from the camera calibration files.
    If that fails, it falls back to the value in the config file.
    """
    try:
        calib = load_json(config.CALIB_PATH)
        extr = load_json(config.EXTR_PATH)

        K = np.array(calib["camera_matrix"], dtype=np.float64)
        dist = np.array(calib["dist_coeffs"], dtype=np.float64).ravel()
        
        rvec = np.array(extr["rvec"], dtype=np.float64).reshape(3, 1)
        tvec = np.array(extr["tvec"], dtype=np.float64).reshape(3, )
        R, _ = cv2.Rodrigues(rvec)
        t = tvec

        n_c, d_c = compute_camera_plane(R, t)

        # Calculate mm_per_pixel at the center of the image
        p1 = pixel_to_world_using_camera_plane(config.FRAME_W / 2, config.FRAME_H / 2, K, dist, R, t, n_c, d_c)
        p2 = pixel_to_world_using_camera_plane(config.FRAME_W / 2 + 1, config.FRAME_H / 2, K, dist, R, t, n_c, d_c)

        if p1 is not None and p2 is not None:
            mm_per_pixel = np.linalg.norm(p1 - p2) * 1000.0  # Convert meters to mm
            print(f"✅ Calculated conversion factor from calibration: {mm_per_pixel:.4f} mm per pixel")
            return mm_per_pixel
        else:
            raise Exception("Could not calculate mm_per_pixel from calibration.")

    except Exception as e:
        print(f"❌ Failed to load calibration and calculate mm/pixel: {e}")
        print("Falling back to default mm_per_pixel from config.")
        return config.MM_PER_PIXEL
