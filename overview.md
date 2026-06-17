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

This document reflects the current runtime behavior in `main.py` as of 2026-06-17.

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
- `perform_reset()`: pause schedulers, reset runtime state, clear queued jobs, insert DB reset row, and send serial reset command
- `clear_capture_queue(...)`: removes pending pre-reset capture work
- `reset_runtime_state(...)`: resets runtime distance, buffers, raw histories, and increments `reset_generation`

Reset-related state:
- `state_lock`: protects shared distance/reset state across threads
- `reset_in_progress`: pauses serial and fallback schedulers during reset
- `reset_generation`: tags capture jobs so stale pre-reset jobs cannot write after reset

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

Startup seeding behavior:
- `get_recent_valid_measurements(...)` fetches recent non-null positive stitch/seam values.
- It no longer filters startup buffer rows by ideal stitch/seam ranges.
- `main.py` loads positive DB values into the runtime buffers and initializes `last_avg_stitch_length_mm` from the stitch buffer mean when available.
- If startup detects a new day, it inserts a reset row and initializes runtime total distance directly to `0.0` after the reset succeeds. It does not immediately re-read `total_distance` from DB on that reset path.

Observed behavior:
- If DB host cannot resolve/connect, inserts are skipped and errors are logged.

### `serial_communicator.py`
Responsibilities:
- Open ESP32 serial port (configured + discovery fallback)
- Read integer stitch counts from newline-delimited serial stream
- Convert stitch deltas to distance using `last_avg_stitch_length_mm`
- Send reset command `R`

Distance formula:

```text
current_total_distance += stitch_delta * last_avg_stitch_length_mm
```

If `last_avg_stitch_length_mm` is `0.0` or unavailable, the distance update is skipped and total distance remains unchanged.

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
- Every queued job stores the current `reset_generation`.
- If reset happens before a queued job finishes, the job's generation no longer matches the current generation and its result is skipped.

### Worker lag signal
- Worker logs warning when queued job wait exceeds one interval:
  - `[WARN] Capture job lag: X.XXs`

## 6. Measurement, Buffer, and DB Rules

Per frame, `process_fabric_immediate(...)`:
1. Capture frame
2. Run inference and compute raw stitch length and seam allowance
3. Add raw camera values to `raw_stitch_history` and `raw_seam_history`
4. Use buffer means when a measurement is missing, or when an out-of-range value is not yet confirmed
5. Apply sustained-confirmation override for repeated out-of-range values
6. Add the final accepted positive values to the runtime buffers
7. Update `serial_communicator.last_avg_stitch_length_mm` from the final accepted stitch length
8. Insert the final accepted values to DB when serial movement occurred and the job is not stale after reset

Important:
- The raw history stores current camera values, including out-of-range values.
- The runtime buffers store final accepted positive values. These can come from DB startup seeding, buffer fallback, normal camera values, or confirmed override values.
- `confirmed_override = True` means the repeated out-of-range camera value is trusted. It is added to the buffer, used for DB insert, and used as the next serial distance multiplier.
- `last_avg_stitch_length_mm` is the multiplier used to convert ESP32 stitch-count deltas into total distance.
- Missing final values can still cause DB insert skip depending on DB manager rules.
- Jobs queued before a reset cannot insert after reset because `process_fabric_immediate(...)` compares the job's `reset_generation` with the current generation.

## 7. Reset and Total Distance Safety

### New-day startup reset
On startup, `main.py` checks the latest DB measurement date.

If the latest DB date is before the current local date:
1. `database_manager.reset_total_distance_on_startup()` inserts a reset row.
2. `reset_runtime_state(serial_communicator)` resets in-memory distance and buffers.
3. Runtime `serial_communicator.current_total_distance` is initialized to `0.0`.
4. The startup path skips the normal DB total-distance reload.

This prevents a stale DB row from restoring an old distance immediately after a successful startup reset.

If no startup reset is needed, `main.py` loads the last DB `total_distance` and continues from that value.

### Manual MQTT reset
`perform_reset()` treats reset as a runtime state boundary:
1. Set `reset_in_progress` so serial/fallback schedulers pause.
2. Acquire `processing_lock` so no active frame can insert while the reset row is being written.
3. Call `reset_runtime_state(...)`.
4. Call `clear_capture_queue(...)` to drop queued pre-reset jobs.
5. Insert DB reset row `(0, 0, 0)`.
6. Send serial command `R` to the ESP32.
7. Wait `RESET_POST_DELAY_SEC`.
8. Clear `reset_in_progress` so schedulers resume.

### Stale job protection
Each capture job stores:

```text
reset_generation = current reset_generation at enqueue time
```

When reset occurs, `reset_runtime_state(...)` increments the global `reset_generation`.

If an old job finishes after reset:

```text
job_reset_generation != current_reset_generation
```

The worker skips the result before DB insert, preventing an old `total_distance` from being written after a reset row.

## 8. Current Config Surface

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

## 9. 15 FPS Recommendations (Applied)

For a 15 FPS camera (~66.7ms/frame):
- `camera.flush_frames: 1`
- `processing.capture_queue_maxsize: 1`

Rationale:
- Large flush counts add capture latency
- Queue size 1 prevents multi-second backlog accumulation

## 10. Known Failure Modes and Meaning

### `Waiting for first serial stitch count...`
- Serial connected but no valid integer stitch line received yet.
- Distance stays at baseline until first count arrives.

### `Capture queue full - dropped oldest pending job`
- Processing is slower than scheduler rate.
- System is preserving freshness (by design), not processing every historical tick.

### `Skipping stale capture result from before reset`
- A capture job was queued before reset and completed after reset.
- The result is intentionally discarded so it cannot insert stale distance into DB.

### `Can't connect to MySQL ... Temporary failure in name resolution`
- DNS/network issue to DB host, not inference/camera logic.
- Inserts are skipped until connectivity is restored.

### `Unable to obtain measurements from frame`
- Model/vision path returned no valid measurement for that frame.
- Can happen due to lighting, framing, motion blur, occlusion, or detection miss.
- If runtime buffers are populated, missing values can be replaced with buffer means before DB insert.

### `Avg stitch length not available yet; skipping distance update`
- ESP32 provided a stitch delta, but `last_avg_stitch_length_mm` is still missing or `0.0`.
- This normally happens before DB startup seeding succeeds or before the first final accepted stitch length is produced.
- Once a positive final stitch length is accepted, `main.py` updates `last_avg_stitch_length_mm` and later serial deltas can update total distance.

## 11. Startup and Run

1. Ensure `.env` has valid DB/MQTT credentials
2. Ensure ESP32 appears on configured serial port (or discoverable)
3. Ensure camera device is accessible (`/dev/video*` permissions)
4. Run:

```bash
python main.py
```

## 12. Shutdown

- SIGINT (`Ctrl+C`) sets `shutdown_event`
- Threads join with timeout
- DB, camera, and serial resources are closed
