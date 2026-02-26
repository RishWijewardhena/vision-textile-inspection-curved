# Stitch V2 Fabric Inspection System

This project is a real-time fabric inspection system that uses computer vision to detect defects in stitches and fabric. It measures stitch length, distance to edge, and identifies consecutive defects.

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

2.  **`config.yaml`:** This file contains the main configuration for the application, including camera settings, serial port, and defect thresholds.

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
*   `serial_communicator.py`: Handles serial communication with the Arduino.
*   `calibration.py`: Contains functions for camera calibration and coordinate system transformations.
*   `cleanup.py`: A thread that cleans up old images from the output directory.
*   `mqtt_heartbeat.py`: A thread that sends MQTT heartbeats to a broker.

## Usage

To run the fabric inspection system, execute the main script:

```bash
python3 main.py
```

The system will start, initialize the camera and serial communication, and begin processing the fabric according to the settings in `config.yaml`.
