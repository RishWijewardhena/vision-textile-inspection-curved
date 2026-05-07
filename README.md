# Vision Textile Inspection - Curved
### Real-time stitch and seam analytics with camera + AI

This project is a real-time fabric inspection system that uses computer vision to measure stitch length and seam allowance, log results to MySQL, and publish MQTT status.

## Project Overview

The system captures frames from a USB camera, runs a YOLO model to detect stitches and fabric edges, converts pixel measurements to millimeters using calibration data, and logs results to MySQL. It also publishes MQTT status and supports a serial-connected ESP32 for stitch count and distance tracking.

## Output Measurements

Each processed frame produces the following outputs:

- **Stitch length (mm)**: Average distance between consecutive stitches detected in the frame.
- **Seam allowance (mm)**: Average distance between stitch positions and the fabric edge.
- **Total distance (mm)**: Running distance derived from ESP32 stitch counts and the latest valid stitch length.

The annotated image for each frame is saved under `output.directory` in [config.yaml](config.yaml). Measurements are inserted into the MySQL table defined by `DB_TABLE` in your `.env` file.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd stitch_v2
    ```

2.  **Install dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

## Configuration

1.  **`.env` file:** Create a `.env` file in the root directory to store your database and MQTT credentials. You can use the `.env.example` file as a template.

2.  **`config.yaml`:** This file contains the main configuration for the application, including camera settings, serial port, and thresholds.

3.  **Camera Calibration:**
    *   `camera_calibration.json`: Contains the camera intrinsic matrix and distortion coefficients.
    *   `camera_extrinsics.json`: Contains the camera extrinsics (rotation and translation vectors) relative to the scene.

    These files are essential for accurate measurements.

## Project Structure

The project is structured into the following modules:

*   `main.py`: The main entry point of the application. It initializes all the components and starts the threads.
*   `config.py`: Loads and provides all configuration from `config.yaml` and `.env` file.
*   `camera_manager.py`: Manages the camera, including initialization and frame capture.
*   `image_processor.py`: Contains the core logic for image processing, including defect detection.
*   `database_manager.py`: Manages the connection and data insertion into the MySQL database.
*   `serial_communicator.py`: Handles serial communication with the ESP32.
*   `calibration.py`: Contains functions for camera calibration and coordinate system transformations.
*   `cleanup.py`: A thread that cleans up old images from the output directory.
*   `mqtt_heartbeat.py`: A thread that sends MQTT heartbeats to a broker.
*   `utils/resource_discovery.py`: Auto-discovers the camera and ESP32 device paths.
*   `scripts/grant_passwordless_sudoers_for_modprobe.sh`: Installs a sudoers drop-in for passwordless `modprobe` commands.

## Usage

To run the fabric inspection system, execute the main script:

```bash
python3 main.py
```

The system will start, initialize the camera and serial communication, and begin processing according to the settings in [config.yaml](config.yaml).

## Runtime Behavior

- **Camera capture**: Frames are captured from the first available `/dev/video*` device that opens successfully. Capture uses the resolution configured in [config.yaml](config.yaml).
- **Reconnect attempts**: When capture fails, the camera is reinitialized. After `camera.max_reconnect_attempts` failures, the app calls `reload_camera()` and retries.
- **Serial-triggered processing**: When the ESP32 serial port is available, processing runs on a timed interval and distance updates are derived from stitch counts.
- **Fallback capture (no serial)**: If the serial port is not available, the app still captures and processes frames on the same `processing.capture_interval`.
- **Image cleanup**: Images under `output.directory` are deleted after `output.image_retention_seconds`.

## MQTT

- **Heartbeat**: Publishes `on` to `machine/<DEVICE_ID>/status/heartbeat` every `MQTT_HEARTBEAT_INTERVAL` seconds.
- **Reset command**: Listens on `machine/<DEVICE_ID>/commands/reset` for `reset` and replies with `reset_success`.
- **Camera issue**: Publishes `issue` to `machine/<DEVICE_ID>/status/camera_issue` when frame capture fails.

## Camera Calibration

- Calibration files are read on startup by `ImageProcessor` via `get_mm_per_pixel()`.
- If calibration files cannot be loaded, the system falls back to `units.mm_per_pixel` from [config.yaml](config.yaml).

## Optional: Passwordless modprobe for camera recovery

If your service runs as a non-root user and you want to allow passwordless `modprobe` for camera recovery, run the helper script as root:

```bash
sudo bash scripts/grant_passwordless_sudoers_for_modprobe.sh
```

This installs a sudoers drop-in that allows the invoking user to run:

- `modprobe -r cdc_acm`
- `modprobe cdc_acm`
- `modprobe -r usb_storage`
- `modprobe usb_storage`

You can verify it with:

```bash
sudo visudo -c -f /etc/sudoers.d/thread-modprobe
sudo cat /etc/sudoers.d/thread-modprobe
```
