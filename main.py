# main.py

import time
import os
import signal
import sys
import threading
from datetime import datetime
from collections import deque

from ultralytics import YOLO
import cv2

import config
import subprocess
from camera_manager import CameraManager
from image_processor import ImageProcessor
from database_manager import DatabaseManager
from serial_communicator import SerialCommunicator
from cleanup import image_cleanup_thread
from mqtt_heartbeat import MqttHeartbeat

# Globals for state management
shutdown_event = threading.Event()
processing_lock = threading.Lock()
last_capture_time = 0
last_processed_distance = 0.0

# Keep the last 5 valid measurements for real-data fallback (no random values).
stitch_length_buffer = deque(maxlen=5)
seam_allowance_buffer = deque(maxlen=5)

# Track raw measurements (in and out of range) for confirmed override detection
raw_stitch_history = deque(maxlen=config.CONFIRM_CONSECUTIVE)
raw_seam_history = deque(maxlen=config.CONFIRM_CONSECUTIVE)

# Session folder for this run
SESSION_FOLDER = None
camera_issue_active = False


def log_ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def is_stitch_length_in_ideal_range(value_mm):
    """Return True when stitch length is inside configured ideal limits."""
    if value_mm is None:
        return False
    return config.IDEAL_STITCH_LENGTH_MM_MIN <= value_mm <= config.IDEAL_STITCH_LENGTH_MM_MAX


def is_seam_allowance_in_ideal_range(value_mm):
    """Return True when seam allowance is inside configured ideal limits."""
    if value_mm is None:
        return False
    return config.IDEAL_SEAM_ALLOWANCE_MM_MIN <= value_mm <= config.IDEAL_SEAM_ALLOWANCE_MM_MAX


def sigint_handler(sig, frame):
    print('Interrupted - shutting down threads...')

    # Trigger graceful shutdown; let main loop close resources in order.
    shutdown_event.set()


signal.signal(signal.SIGINT, sigint_handler)

def reload_camera():
    """Reload webcam driver (uvcvideo)."""
    print(log_ts() + " 🔄 Reloading webcam driver...")
    try:
        subprocess.run(["sudo", "modprobe", "-r", "uvcvideo"], check=True, capture_output=True)
        time.sleep(0.5)  # Give system time to unload
        subprocess.run(["sudo", "modprobe", "uvcvideo"], check=True, capture_output=True)
        time.sleep(0.5)  # Give driver time to load
        print(log_ts() + " ✅ Webcam driver reloaded")
    except subprocess.CalledProcessError as e:
        print(log_ts() + f" ⚠️ Failed to reload webcam driver: {e}")
    except PermissionError:
        print(log_ts() + f" ❌ Permission denied: Run 'sudo ./scripts/grant_passwordless_sudoers_for_modprobe.sh' first")


def process_fabric_immediate(
    image_processor,
    camera_manager,
    serial_communicator,
    db_manager,
    session_output_dir,
    delta_stitches,
    heartbeat,
    skip_db_insert=False,
):
    """
    Process fabric immediately when triggered and INSERT ONCE per processed frame.
    Also updates SerialCommunicator with the latest measured stitch length
    to improve distance calculation.
    """

    if not processing_lock.acquire(blocking=False):
        print("⚠️ WARNING: Processing lock in use - skipping capture")
        return

    global camera_issue_active

    try:
        capture_ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        print(f" Starting fabric analysis at {capture_ts}")

        frame = camera_manager.capture_frame_safely()
        if frame is None:
            print("❌ Could not capture frame - reporting camera issue")
            image_processor.process_frame(
                None,
                serial_communicator.current_total_distance,
            )

            if heartbeat:
                try:
                    heartbeat.client.publish(
                        config.MQTT_CAMERA_ISSUE_TOPIC,
                        payload="issue",
                        qos=0,
                        retain=False,
                    )
                    print(log_ts() + f" ! MQTT camera issue sent: {config.MQTT_CAMERA_ISSUE_TOPIC} -> issue")
                except Exception as exc:
                    print(log_ts() + f" ⚠️ MQTT camera issue publish failed: {exc}")

            camera_issue_active = True
            return

        print("✅ Frame captured, starting AI inference...")
        start_time = time.time()

        annotated, summary, result = image_processor.process_frame(
            frame,
            serial_communicator.current_total_distance
        )

        processing_time = time.time() - start_time

        # ✅ Update serial distance model with latest measured stitch length (ONLY if valid)
        avg_len = summary.get("avg_stitch_length_mm")
        if is_stitch_length_in_ideal_range(avg_len):
            serial_communicator.last_avg_stitch_length_mm = float(avg_len)

        out_path = os.path.join(session_output_dir, f"fabric_{capture_ts}.jpg")
        cv2.imwrite(out_path, annotated)

        print(f"!! FABRIC ANALYSIS RESULTS ({summary['timestamp']}):")
        print(f"   ├─ Total Distance: {summary['total_distance_mm']:.2f}mm")
        print(f"   ├─ Total Edges: {summary['edge_count']}")

        if summary.get('avg_stitch_length_mm') is not None:
            print(f"   ├─ Avg Stitch Length: {summary['avg_stitch_length_mm']:.2f}mm")
        else:
            print("   ├─ Stitch Length: Not measurable")

        if summary.get('avg_distance_mm') is not None:
            print(f"   ├─ Avg Stitch-Top Edge Distance: {summary['avg_distance_mm']:.2f}mm")
        else:
            print("   ├─ Avg Stitch-Top Edge Distance: Not measurable")

        print(f"   └─ Processing Time: {processing_time:.2f}s")

        # ✅ Insert into MySQL ON EVERY processed frame (no periodic inserts)
        stitch_length = summary.get("avg_stitch_length_mm")          # avg_length -> stitch_length
        seam_allowance = summary.get("avg_distance_mm")              # avg_dist -> seam_allowance
        total_distance = serial_communicator.current_total_distance  # total_distance

        # Track ALL raw measurements (in and out of range) for confirmed override detection
        if stitch_length is not None:
            raw_stitch_history.append(stitch_length)
        if seam_allowance is not None:
            raw_seam_history.append(seam_allowance)

        # Keep a rolling buffer of valid real measurements.
        if is_stitch_length_in_ideal_range(stitch_length):
            stitch_length_buffer.append(float(stitch_length))
        if is_seam_allowance_in_ideal_range(seam_allowance):
            seam_allowance_buffer.append(float(seam_allowance))

        # If we moved forward (delta>0) but this frame missed a measurement,
        # fall back to mean of recent valid real measurements.
        if delta_stitches > 0:
            if stitch_length is None and stitch_length_buffer:
                stitch_length = sum(stitch_length_buffer) / len(stitch_length_buffer)
                print(
                    f"ℹ️ Using buffered stitch_length mean: {stitch_length:.3f}mm "
                    f"from {len(stitch_length_buffer)} samples"
                )

            if seam_allowance is None and seam_allowance_buffer:
                seam_allowance = sum(seam_allowance_buffer) / len(seam_allowance_buffer)
                print(
                    f"ℹ️ Using buffered seam_allowance mean: {seam_allowance:.3f}mm "
                    f"from {len(seam_allowance_buffer)} samples"
                )

        confirmed_override = False

        # Stitch length: Check for N consecutive similar samples above soft upper limit
        if stitch_length is not None and not is_stitch_length_in_ideal_range(stitch_length):
            if stitch_length > config.IDEAL_STITCH_LENGTH_MM_MAX:
                recent = [v for v in list(raw_stitch_history) if v is not None]
                if len(recent) >= config.CONFIRM_CONSECUTIVE and all(v > config.IDEAL_STITCH_LENGTH_MM_MAX - config.CONFIRM_TOLERANCE_MM for v in recent):
                    confirmed_override = True
                    print(log_ts() + f" **** Stitch length above {config.IDEAL_STITCH_LENGTH_MM_MAX}mm but sustained for {config.CONFIRM_CONSECUTIVE} samples - accepting as valid")
            elif stitch_length < config.IDEAL_STITCH_LENGTH_MM_MIN:
                recent = [v for v in list(raw_stitch_history) if v is not None]
                if len(recent) >= config.CONFIRM_CONSECUTIVE and all(v < config.IDEAL_STITCH_LENGTH_MM_MIN + config.CONFIRM_TOLERANCE_MM for v in recent):
                    confirmed_override = True
                    print(log_ts() + f" **** Stitch length below {config.IDEAL_STITCH_LENGTH_MM_MIN}mm but sustained for {config.CONFIRM_CONSECUTIVE} samples - accepting as valid")

        # Seam allowance: Check for N consecutive similar samples above soft upper limit
        if seam_allowance is not None and not is_seam_allowance_in_ideal_range(seam_allowance):
            if seam_allowance > config.IDEAL_SEAM_ALLOWANCE_MM_MAX:
                recent_w = [v for v in list(raw_seam_history) if v is not None]
                if len(recent_w) >= config.CONFIRM_CONSECUTIVE and all(v > config.IDEAL_SEAM_ALLOWANCE_MM_MAX - config.CONFIRM_TOLERANCE_MM for v in recent_w):
                    confirmed_override = True
                    print(log_ts() + f" **** Seam allowance above {config.IDEAL_SEAM_ALLOWANCE_MM_MAX}mm but sustained for {config.CONFIRM_CONSECUTIVE} samples - accepting as valid")
            elif seam_allowance < config.IDEAL_SEAM_ALLOWANCE_MM_MIN:
                recent_w = [v for v in list(raw_seam_history) if v is not None]
                if len(recent_w) >= config.CONFIRM_CONSECUTIVE and all(v < config.IDEAL_SEAM_ALLOWANCE_MM_MIN + config.CONFIRM_TOLERANCE_MM for v in recent_w):
                    confirmed_override = True
                    print(log_ts() + f" **** Seam allowance below {config.IDEAL_SEAM_ALLOWANCE_MM_MIN}mm but sustained for {config.CONFIRM_CONSECUTIVE} samples - accepting as valid")

        # Final safety filter before persistence: reject out-of-range values UNLESS confirmed by override
        if stitch_length is not None and not is_stitch_length_in_ideal_range(stitch_length) and not confirmed_override:
            print(
                "!!! Ignoring out-of-range stitch_length before DB insert: "
                f"{stitch_length:.3f}mm "
                f"(allowed {config.IDEAL_STITCH_LENGTH_MM_MIN:.3f}-"
                f"{config.IDEAL_STITCH_LENGTH_MM_MAX:.3f}mm)"
            )
            stitch_length = None

        if seam_allowance is not None and not is_seam_allowance_in_ideal_range(seam_allowance) and not confirmed_override:
            print(
                "!!! Ignoring out-of-range seam_allowance before DB insert: "
                f"{seam_allowance:.3f}mm "
                f"(allowed {config.IDEAL_SEAM_ALLOWANCE_MM_MIN:.3f}-"
                f"{config.IDEAL_SEAM_ALLOWANCE_MM_MAX:.3f}mm)"
            )
            seam_allowance = None


        ok = False
        if not skip_db_insert:
            ok = db_manager.insert_measurement(
                stitch_length=stitch_length,
                seam_allowance=seam_allowance,
                total_distance=total_distance
            )

            if ok:
                print("✅ MySQL insert done (per-frame)")
            else:
                print(" MySQL insert skipped (per-frame)")
        else:
            print("ℹ️ Skipping DB insert (fallback mode - no ESP32 connection)")

    except Exception as e:
        print(f"❌ ERROR in fabric processing: {e}")

    finally:
        processing_lock.release()




def serial_monitor_thread(serial_communicator, image_processor, camera_manager, db_manager, session_output_dir, heartbeat):
    """
    Thread that monitors serial for stitch count / distance updates
    and triggers image processing when distance change criteria are met.
    """
    global last_capture_time, last_processed_distance

    print("[INFO] Serial monitor thread started, reading distance data...")

    previous_stitch_count = serial_communicator.read_serial_data()
    pending_delta_stitches = 0
    waiting_log_last_time = 0.0

    while not shutdown_event.is_set():
        try:
            last_stitch_count = serial_communicator.read_serial_data()
            if previous_stitch_count is None:
                now = time.time()
                if now - waiting_log_last_time >= 2.0:
                    print("[INFO] Waiting for first serial stitch count...")
                    waiting_log_last_time = now

                if last_stitch_count is not None:
                    previous_stitch_count = last_stitch_count
                    print(f"[INFO] First serial stitch count received: {previous_stitch_count}")
            elif last_stitch_count is not None:
                delta = last_stitch_count - previous_stitch_count
                previous_stitch_count = last_stitch_count

                # Update total distance based on the latest stitch count and AI stitch length
                if delta >= 0:
                    pending_delta_stitches += delta
                    serial_communicator.update_distance_from_stitch_count(delta)
                else:
                    print(f"[WARN] Stitch count decreased ({delta}); ignoring this sample")

            current_time = time.time()

            if current_time - last_capture_time >= config.CAPTURE_INTERVAL:
                print(
                    f"\n=== FABRIC PROCESSING TRIGGERED "
                    f"(Distance: {serial_communicator.current_total_distance:.2f}mm, "
                    f"Interval: {config.CAPTURE_INTERVAL:.2f}s) ==="
                )

                processing_thread = threading.Thread(
                    target=process_fabric_immediate,
                    args=(
                        image_processor,
                        camera_manager,
                        serial_communicator,
                        db_manager,
                        session_output_dir,
                        pending_delta_stitches,
                        heartbeat,
                    ),
                    daemon=True
                )
                processing_thread.start()

                last_capture_time = current_time
                last_processed_distance = serial_communicator.current_total_distance
                pending_delta_stitches = 0


            time.sleep(0.01)

        except Exception as e:
            print(f"[ERROR] Serial monitor thread: {e}")
            shutdown_event.set()


def fallback_capture_thread(
    image_processor,
    camera_manager,
    serial_communicator,
    db_manager,
    session_output_dir,
    heartbeat,
    stop_event,
):
    """Capture and process frames on a timer when serial input is unavailable."""
    global last_capture_time

    print("[INFO] Fallback capture thread started (serial unavailable)")

    while not shutdown_event.is_set() and not stop_event.is_set():
        try:
            current_time = time.time()
            if current_time - last_capture_time >= config.CAPTURE_INTERVAL:
                print(
                    f"\n=== FALLBACK CAPTURE TRIGGERED "
                    f"(Distance: {serial_communicator.current_total_distance:.2f}mm, "
                    f"Interval: {config.CAPTURE_INTERVAL:.2f}s) ==="
                )

                # Publish ESP32 issue to MQTT
                if heartbeat:
                    try:
                        heartbeat.publish_esp32_issue()
                    except Exception as exc:
                        print(f"⚠️ MQTT ESP32 issue publish failed: {exc}")

                processing_thread = threading.Thread(
                    target=process_fabric_immediate,
                    args=(
                        image_processor,
                        camera_manager,
                        serial_communicator,
                        db_manager,
                        session_output_dir,
                        0,
                        heartbeat,
                        True,
                    ),
                    daemon=True,
                )
                processing_thread.start()

                last_capture_time = current_time

            time.sleep(0.05)
        except Exception as e:
            print(f"[ERROR] Fallback capture thread: {e}")
            shutdown_event.set()

    if stop_event.is_set() and not shutdown_event.is_set():
        print("[INFO] Fallback capture thread stopped (serial available)")


def main():
    """Main function to start the system"""

    mqtt_reset_topic = getattr(
        config,
        "MQTT_RESET_TOPIC",
        f"machine/{config.DEVICE_ID}/control/reset",
    )
    reset_post_delay_sec = getattr(config, "RESET_POST_DELAY_SEC", 0.5)

    # Create timestamped output folder for this session
    session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_output_dir = os.path.join(config.OUTPUT_DIR, session_timestamp)
    os.makedirs(session_output_dir, exist_ok=True)
    print(f"📁 Session output directory: {session_output_dir}")

    # Initialize MQTT heartbeat
    heartbeat = None
    reset_requested = threading.Event()

    def queue_reset_request():
        """Queue reset work to run inside the main loop thread."""
        reset_requested.set()

    def perform_reset():
        """Reset DB values, ESP32 count, and runtime smoothing state."""
        print("🔁 Processing reset command...")

        db_success = False
        if db_manager:
            db_success = db_manager.insert_measurement(
                total_distance=0.0,
                stitch_length=0.0,
                seam_allowance=0.0,
                ignore_limits=True,
            )
            if db_success:
                print("✅ DB reset row inserted (0,0,0)")
            else:
                print("⚠️ DB reset row insert failed")
        else:
            print("⚠️ DB unavailable for reset row insert")

        serial_success = False
        if serial_communicator:
            serial_success = serial_communicator.send_command("R")
            if serial_success:
                print("✅ Serial reset command sent: R")
            else:
                print("⚠️ Serial reset command failed")
        else:
            print("⚠️ Serial reader unavailable for reset command")
        

        # Give ESP32 time to apply reset before using stitch count baseline again.
        time.sleep(reset_post_delay_sec)

        serial_communicator.current_total_distance = 0.0
        serial_communicator.last_avg_stitch_length_mm = 0.0
        stitch_length_buffer.clear()
        seam_allowance_buffer.clear()
        print("✅ Runtime counters and buffers reset")

        if db_success and serial_success and heartbeat:
            heartbeat.publish_reset_success()
            print(f"✅ MQTT reset acknowledgment published: {mqtt_reset_topic} -> reset_success")

    
    try:
        heartbeat = MqttHeartbeat(
            broker=config.MQTT_SERVER,
            port=config.MQTT_PORT,
            username=config.MQTT_USERNAME,
            password=config.MQTT_PASSWORD,
            topic=config.MQTT_HEARTBEAT_TOPIC,
            interval_sec=config.MQTT_HEARTBEAT_INTERVAL,
            tls_insecure=config.MQTT_TLS_INSECURE,
            reset_topic=mqtt_reset_topic,
            on_reset=queue_reset_request,
            esp32_issue_topic=config.MQTT_ESP32_ISSUE_TOPIC,
        )
        heartbeat.start()
        print(f"✅ MQTT heartbeat started: {config.MQTT_HEARTBEAT_TOPIC} (every {config.MQTT_HEARTBEAT_INTERVAL}s)")
    except Exception as e:
        print(f"⚠️ MQTT heartbeat not started: {e} (continuing without heartbeat)")

    

    print("🚀 STARTING OPTIMIZED FABRIC INSPECTION SYSTEM")
    print("=" * 50)
    print(f"Data will be inserted into MySQL at {config.DB_CONFIG['host']}/{config.DB_CONFIG['database']}")
    print(f"Images in {config.OUTPUT_DIR} will be deleted after {config.IMAGE_RETENTION_SECONDS/3600:.1f} hours")
    print("=" * 50)
    
    print("intemediate delay for smooth restart...")
    time.sleep(1.0)

    # Initialize components
    print("🤖 Loading AI model...")
    model = YOLO("best_curve_100.pt")
    model.to(config.DEVICE)
    print(f"✅ Model loaded on {config.DEVICE}")

    camera_manager = CameraManager(reload_callback=reload_camera)
    if not camera_manager.cap:
        print("⚠️ Camera initialization failed; system will keep running and report camera issues until feed recovers")

    image_processor = ImageProcessor(model)
    db_manager = DatabaseManager()
    # Initialize SerialCommunicator with the current total distance from DB (or 0.0 if not available)        
    serial_communicator = SerialCommunicator()



    #reset the total distnace to 0 on startup to avoid false triggers
    if db_manager:
        last_date=db_manager.get_last_measurement_date()
        print(f"📅 Last measurement date in DB: {last_date}")
        today_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"📅 Current system date: {today_str}")

        if last_date and last_date[:10] < today_str[:10]:  # Compare only YYYY-MM-DD
            print("🔄 Resetting total distance on startup...")
            try:
                db_manager.reset_total_distance_on_startup()
            except Exception as e:
                print(f"❌ Failed to reset total distance: {e}")

        elif last_date== "No records found":
            print("⚠️ No records found in DB adding the first row")
            db_manager.reset_total_distance_on_startup()

        elif last_date==None:
            print("⚠️ Could not retrieve last measurement date - skipping total distance reset")
            
        else:
            print("✅ Total distance reset not needed on startup")

        # retrieving the last total distance from the database to initialize counting session with the correct value
        last_total_distance = db_manager.get_last_total_distance()
        if last_total_distance is not None:
            serial_communicator.current_total_distance = last_total_distance
            print(f"🔄 Initialized total distance from DB: {last_total_distance:.2f}mm")

        # Seed fallback buffers with recent real measurements from DB.
        recent_real = db_manager.get_recent_valid_measurements(limit=5)
        for stitch_val, seam_val in recent_real:
            if is_stitch_length_in_ideal_range(stitch_val):
                stitch_length_buffer.append(stitch_val)
            if is_seam_allowance_in_ideal_range(seam_val):
                seam_allowance_buffer.append(seam_val)
        if recent_real:
            print(
                f"🔄 Seeded measurement buffers from DB: {len(recent_real)} samples "
                f"(stitch mean {sum(stitch_length_buffer)/len(stitch_length_buffer):.3f}mm, "
                f"seam mean {sum(seam_allowance_buffer)/len(seam_allowance_buffer):.3f}mm)"
            )



    threads = []
    serial_thread = None
    fallback_thread = None
    fallback_stop_event = threading.Event()
    fallback_active = False
    serial_active = False
    last_serial_probe_time = 0.0
    serial_probe_interval_sec = 1.0

    if serial_communicator.serial_port is not None:
        serial_thread = threading.Thread(
            target=serial_monitor_thread,
            args=(serial_communicator, image_processor, camera_manager, db_manager, session_output_dir, heartbeat),
            daemon=True,
        )
        serial_thread.start()
        threads.append(serial_thread)
        serial_active = True
        print("✅ Serial monitor thread started")
    else:
        print("⚠️ Serial monitor thread not started: Serial port not available.")
        fallback_thread = threading.Thread(
            target=fallback_capture_thread,
            args=(
                image_processor,
                camera_manager,
                serial_communicator,
                db_manager,
                session_output_dir,
                heartbeat,
                fallback_stop_event,
            ),
            daemon=True,
        )
        fallback_thread.start()
        threads.append(fallback_thread)
        fallback_active = True
        print("✅ Fallback capture thread started")

    cleanup_thread = threading.Thread(
        target=image_cleanup_thread,
        args=(shutdown_event, session_output_dir),
        daemon=True
    )
    cleanup_thread.start()
    threads.append(cleanup_thread)
    print("✅ Image cleanup thread started")

    print("🎯 System ready! Processing fabric...")
    print("-" * 50)

    try:
        while not shutdown_event.is_set():
            # print("⏳ Main thread idle - waiting for serial triggers...")
            if reset_requested.is_set():
                reset_requested.clear()
                perform_reset()

            if fallback_active and not serial_active:
                now = time.time()
                if now - last_serial_probe_time >= serial_probe_interval_sec:
                    last_serial_probe_time = now
                    serial_communicator.read_serial_data()
                    if serial_communicator.serial_port is not None:
                        print("✅ Serial port detected; switching to serial monitor thread")
                        fallback_stop_event.set()
                        if fallback_thread:
                            fallback_thread.join(timeout=2.0)
                        serial_thread = threading.Thread(
                            target=serial_monitor_thread,
                            args=(
                                serial_communicator,
                                image_processor,
                                camera_manager,
                                db_manager,
                                session_output_dir,
                                heartbeat,
                            ),
                            daemon=True,
                        )
                        serial_thread.start()
                        threads.append(serial_thread)
                        serial_active = True
                        fallback_active = False

            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n🛑 Shutdown requested...")
        if heartbeat:
            heartbeat.stop()
        shutdown_event.set()

    print("🔄 Waiting for threads to finish...")
    for t in threads:
        t.join(timeout=2.0)

    # Close resources
    db_manager.close()
    camera_manager.release()
    serial_communicator.close()

    print("✅ System shutdown complete")


if __name__ == "__main__":
    main()