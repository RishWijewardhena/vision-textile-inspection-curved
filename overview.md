# THREAD — Fabric Inspection System (Updated Overview)

## 1. Purpose
THREAD is a real-time fabric inspection pipeline that:
- Captures camera frames on a fixed interval (`processing.capture_interval`, default 2.0s)
- Runs YOLO-based detection and geometry measurements
- Produces stitch length and seam allowance (mm)
- Tracks machine travel distance from ESP32 stitch counts
- Writes validated measurements to MySQL
- Publishes health/reset signals via MQTT
- Cleans old images automatically

This document reflects the current runtime behavior in `main.py` as of 2026-05-25.

## 2. Current Runtime Architecture

### Main design
The system now uses a **deterministic scheduler + bounded queue + single processing worker**.

- Scheduler(s):
  - `serial_monitor_thread(...)` when serial is available
  - `fallback_capture_thread(...)` when serial is unavailable
- Both schedulers trigger every `CAPTURE_INTERVAL` using `time.monotonic()`.
- Instead of spawning unlimited processing threads, schedulers enqueue work to `capture_queue`.
- A single `processing_worker_thread(...)` consumes jobs sequentially and runs `process_fabric_immediate(...)`.

### Why this architecture
- Keeps 2-second scheduling stable even when inference/DB are slow
- Prevents thread-churn and lock contention storms
- Keeps memory bounded with queue backpressure

## 3. Main Components

### `main.py`
Responsibilities:
- Bootstrap model, camera, DB, serial, MQTT
- Start worker/scheduler/cleanup threads
- Handle serial-to-fallback and fallback-to-serial transitions
- Handle reset commands from MQTT

Key functions:
- `process_fabric_immediate(...)`: capture -> infer -> filter -> DB insert
- `enqueue_capture_job(...)`: bounded queue writer, drops oldest when full
- `processing_worker_thread(...)`: single consumer
- `serial_monitor_thread(...)`: serial reads + interval scheduling
- `fallback_capture_thread(...)`: timer scheduling when ESP32 missing
- `perform_reset()`: DB reset row + serial reset command + runtime state reset

### `camera_manager.py`
Responsibilities:
- Open and configure V4L2 camera
- Capture robustly with reconnect logic
- Keep frame freshness with small configurable buffer drain

Current capture behavior:
- Uses `grab()` for a small `CAMERA_FLUSH_FRAMES` drain (configurable)
- Then uses a final `read()` for the frame used in inference

### `image_processor.py`
Responsibilities:
- Run YOLO inference
- Compute stitch/edge metrics
- Return annotated frame and summary statistics

### `database_manager.py`
Responsibilities:
- DB connect/reconnect
- Insert rows with safety checks
- Fetch previous rows for startup seeding

Observed behavior:
- If DB host cannot resolve/connect, inserts are skipped and errors are logged.

### `serial_communicator.py`
Responsibilities:
- Open ESP32 serial port (configured + discovery fallback)
- Read integer stitch counts from newline-delimited serial stream
- Convert stitch deltas to distance using last valid AI stitch length
- Send reset command `R`

### `mqtt_heartbeat.py`
Responsibilities:
- Publish heartbeat
- Subscribe for reset commands
- Publish reset success and issue states

### `cleanup.py`
Responsibilities:
- Periodic image retention cleanup excluding active session dir

## 4. Thread Model

Threads started at runtime:
1. `processing_worker_thread` (always)
2. `serial_monitor_thread` (if serial available) OR `fallback_capture_thread` (if serial unavailable)
3. `image_cleanup_thread` (always)
4. MQTT internal thread(s) when heartbeat is enabled

Main thread:
- Handles queued reset requests
- Monitors serial availability to switch fallback -> serial mode

## 5. Capture Scheduling and Backpressure

### Interval logic
- Uses monotonic clock for stable periodic timing
- Each tick enqueues a capture job

### Queue policy
- Queue size controlled by `CAPTURE_QUEUE_MAXSIZE` (default 1)
- If queue is full:
  - oldest pending job is dropped
  - newest job is inserted
- This is **latest-data-first** behavior (real-time freshness over processing every historical tick)

### Worker lag signal
- Worker logs warning when queued job wait exceeds one interval:
  - `[WARN] Capture job lag: X.XXs`

## 6. Measurement and DB Rules

Per frame, `process_fabric_immediate(...)`:
1. Capture frame
2. Run inference and compute summary
3. Update serial conversion model when stitch length is valid
4. Save annotated image
5. Apply range validation
6. Optional fallback to buffer means when movement occurred and direct measurement is missing
7. Insert to DB (unless disabled in fallback mode)

Important:
- Out-of-range values are filtered unless sustained-confirmation override criteria are met.
- Missing real measurements can cause DB insert skip depending on DB manager rules.

## 7. Current Config Surface

### `config.yaml`
- `processing.capture_interval`
- `processing.min_distance_change_mm`
- `processing.capture_queue_maxsize` (now present; recommended `1`)
- `camera.flush_frames` (now present; recommended `1` for 15 FPS)
- camera resolution/index/reconnect attempts
- ideal ranges and class ids

### `config.py` derived vars
- `CAPTURE_INTERVAL`
- `CAPTURE_QUEUE_MAXSIZE`
- `CAMERA_FLUSH_FRAMES`
- DB/MQTT credentials from `.env`

## 8. 15 FPS Recommendations (Applied)

For a 15 FPS camera (~66.7ms/frame):
- `camera.flush_frames: 1`
- `processing.capture_queue_maxsize: 1`

Rationale:
- Large flush counts add capture latency
- Queue size 1 prevents multi-second backlog accumulation

## 9. Known Failure Modes and Meaning

### `Waiting for first serial stitch count...`
- Serial connected but no valid integer stitch line received yet.
- Distance stays at baseline until first count arrives.

### `Capture queue full - dropped oldest pending job`
- Processing is slower than scheduler rate.
- System is preserving freshness (by design), not processing every historical tick.

### `Can't connect to MySQL ... Temporary failure in name resolution`
- DNS/network issue to DB host, not inference/camera logic.
- Inserts are skipped until connectivity is restored.

### `Unable to obtain measurements from frame`
- Model/vision path returned no valid measurement for that frame.
- Can happen due to lighting, framing, motion blur, occlusion, or detection miss.

## 10. Startup and Run

1. Ensure `.env` has valid DB/MQTT credentials
2. Ensure ESP32 appears on configured serial port (or discoverable)
3. Ensure camera device is accessible (`/dev/video*` permissions)
4. Run:

```bash
python main.py
```

## 11. Shutdown

- SIGINT (`Ctrl+C`) sets `shutdown_event`
- Threads join with timeout
- DB, camera, and serial resources are closed

