
import os
import torch
import yaml
from dotenv import load_dotenv

load_dotenv()

# ---------------------------
# Load Configuration from YAML
# ---------------------------
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# ---------------------------
# GPU Configuration
# ---------------------------
DEVICE = torch.device(config['gpu']['device'] if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ---------------------------
# Configuration Parameters
# ---------------------------
SERIAL_PORT = config['serial']['port']
BAUDRATE = config['serial']['baudrate']
CAMERA_IDX = config['camera']['index']
FRAME_W = config['camera']['width']
FRAME_H = config['camera']['height']
MAX_RECONNECT_ATTEMPTS = int(config.get('camera', {}).get('max_reconnect_attempts', 3))
CAMERA_FLUSH_FRAMES = int(config.get('camera', {}).get('flush_frames', 2))
OUTPUT_DIR = config['output']['directory']
os.makedirs(OUTPUT_DIR, exist_ok=True)
IMAGE_RETENTION_SECONDS = config['output']['image_retention_seconds']
CLEANUP_INTERVAL = config['output']['cleanup_interval']

# ---------------------------
# MySQL Database Configuration
# ---------------------------
DB_CONFIG = {
    'host': os.getenv("DB_HOST"),
    'user': os.getenv("DB_USER"),
    'password': os.getenv("DB_PASSWORD"),
    'database': os.getenv("DB_DATABASE")
}
DB_TABLE = os.getenv("DB_TABLE")

# ---------------------------
# MQTT Config (Heartbeat)
# -------------------------
MQTT_SERVER = os.getenv("MQTT_SERVER")
MQTT_PORT = int(os.getenv("MQTT_PORT"))
MQTT_USERNAME = os.getenv("MQTT_USERNAME")
MQTT_PASSWORD = os.getenv("MQTT_PASSWORD")

# device id = DB_TABLE (as you specified)
DEVICE_ID = DB_TABLE
MQTT_HEARTBEAT_TOPIC = f"machine/{DEVICE_ID}/status/heartbeat"
MQTT_CAMERA_ISSUE_TOPIC = f"machine/{DEVICE_ID}/status/camera_issue"
MQTT_ESP32_ISSUE_TOPIC = f"machine/{DEVICE_ID}/status/esp32_issue"
MQTT_HEARTBEAT_INTERVAL = 2.0  # seconds
MQTT_TLS_INSECURE = os.getenv("MQTT_TLS_INSECURE", "true").lower() in ('true', '1', 't')
MQTT_RESET_TOPIC = f"machine/{DEVICE_ID}/commands/reset"
RESET_POST_DELAY_SEC=8.0

# ---------------------------
# Timing Configuration
# ---------------------------
DB_INSERT_INTERVAL = config['database']['insert_interval']
CAPTURE_INTERVAL = config['processing']['capture_interval']
MIN_DISTANCE_CHANGE_MM = config['processing']['min_distance_change_mm']
CAPTURE_QUEUE_MAXSIZE = int(config.get('processing', {}).get('capture_queue_maxsize', 1))

# ---------------------------
# Offsets (from .env)
# ---------------------------
STITCH_LENGTH_OFFSET_MM = float(os.getenv("STITCH_LENGTH_OFFSET_MM", "0"))
SEAM_ALLOWANCE_OFFSET_MM = float(os.getenv("SEAM_ALLOWANCE_OFFSET_MM", "1.0"))

# ---------------------------
# Ideal Ranges
# ---------------------------
IDEAL_SEAM_ALLOWANCE_MM_MIN = config['ideal_ranges']['seam_allowance_mm_min']
IDEAL_SEAM_ALLOWANCE_MM_MAX = config['ideal_ranges']['seam_allowance_mm_max']
IDEAL_STITCH_LENGTH_MM_MIN = config['ideal_ranges']['stitch_length_mm_min']
IDEAL_STITCH_LENGTH_MM_MAX = config['ideal_ranges']['stitch_length_mm_max']

# ---------------------------
# Confirmed Override Parameters
# ---------------------------
CONFIRM_CONSECUTIVE = 4  # number of consecutive out-of-range measurements to treat as valid
CONFIRM_TOLERANCE_MM = 0.75  # mm — how close consecutive out-of-range samples must be to the limit

# ---------------------------
# Units and Classes
# ---------------------------
CALIB_PATH = config['camera']['calib_path']
EXTR_PATH = config['camera']['extr_path']
MM_PER_PIXEL = config['units']['mm_per_pixel']
STITCH_CLASS_ID = config['classes']['stitch']
EDGE_CLASS_ID = config['classes']['edge']

# ---------------------------
# Data Storage
# ---------------------------
MACHINE_ID = config['machine']['id']
