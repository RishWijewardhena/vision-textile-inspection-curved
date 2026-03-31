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

# Session folder for this run
SESSION_FOLDER = None


def sigint_handler(sig, frame):
    print('Interrupted - shutting down threads...')

    # Trigger graceful shutdown; let main loop close resources in order.
    shutdown_event.set()


signal.signal(signal.SIGINT, sigint_handler)


def process_fabric_immediate(
    image_processor,
    camera_manager,
    serial_communicator,
    db_manager,
    session_output_dir,
    delta_stitches,
):
    """
    Process fabric immediately when triggered and INSERT ONCE per processed frame.
    Also updates SerialCommunicator with the latest measured stitch length
    to improve distance calculation.
    """

    if not processing_lock.acquire(blocking=False):
        print("⚠️ WARNING: Processing lock in use - skipping capture")
        return

    try:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        print(f"🔍 Starting fabric analysis at {ts}")

        frame = camera_manager.capture_frame_safely()
        if frame is None:
            print("❌ Could not capture frame - skipping analysis")
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
        if avg_len is not None and avg_len > 0:
            serial_communicator.last_avg_stitch_length_mm = float(avg_len)

        out_path = os.path.join(session_output_dir, f"fabric_{ts}.jpg")
        cv2.imwrite(out_path, annotated)

        print(f"📊 FABRIC ANALYSIS RESULTS ({summary['timestamp']}):")
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

        # Keep a rolling buffer of valid real measurements.
        if stitch_length is not None and stitch_length > 0:
            stitch_length_buffer.append(float(stitch_length))
        if seam_allowance is not None and seam_allowance > 0:
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


        ok = db_manager.insert_measurement(
            stitch_length=stitch_length,
            seam_allowance=seam_allowance,
            total_distance=total_distance
        )

        if ok:
            print("✅ MySQL insert done (per-frame)")
        else:
            print("❌ MySQL insert failed (per-frame)")

        print(f"⚡ ANALYSIS COMPLETE: {processing_time:.2f}s total")

    except Exception as e:
        print(f"❌ ERROR in fabric processing: {e}")

    finally:
        processing_lock.release()




def serial_monitor_thread(serial_communicator, image_processor, camera_manager, db_manager, session_output_dir):
    """
    Thread that monitors serial for stitch count / distance updates
    and triggers image processing when distance change criteria are met.
    """
    global last_capture_time, last_processed_distance

    print("[INFO] Serial monitor thread started, reading distance data...")

    previous_stitch_count = serial_communicator.read_serial_data()
    waiting_log_last_time = 0.0
    bootstrap_done = False

    while not shutdown_event.is_set():
        try:
            if previous_stitch_count is None:
                now = time.time()
                if now - waiting_log_last_time >= 2.0:
                    print("[INFO] Waiting for first serial stitch count...")
                    waiting_log_last_time = now

                # Run one bootstrap frame so AI can provide initial real measurements
                # even before serial starts streaming counts.
                if not bootstrap_done and now - last_capture_time >= config.CAPTURE_INTERVAL:
                    print("[INFO] Running bootstrap processing while waiting for serial data...")
                    processing_thread = threading.Thread(
                        target=process_fabric_immediate,
                        args=(
                            image_processor,
                            camera_manager,
                            serial_communicator,
                            db_manager,
                            session_output_dir,
                            0,
                        ),
                        daemon=True,
                    )
                    processing_thread.start()
                    bootstrap_done = True

                previous_stitch_count = serial_communicator.read_serial_data()
                time.sleep(0.1)
                continue

            last_stitch_count = serial_communicator.read_serial_data()
            if last_stitch_count is None:
                time.sleep(0.01)
                continue

            delta = last_stitch_count - previous_stitch_count
            previous_stitch_count = last_stitch_count

            # Update total distance based on the latest stitch count and AI stitch length
            if delta >= 0:  # Only update if stitch count has increased
                serial_communicator.update_distance_from_stitch_count(delta)

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
                        delta,
                    ),
                    daemon=True
                )
                processing_thread.start()

                last_capture_time = current_time
                last_processed_distance = serial_communicator.current_total_distance

            # If this log is too noisy, you can comment it out
            # else:
            #     print(
            #         f"⚠️ Skipping capture: Time since last capture: {current_time - last_capture_time:.2f}s, "
            #         f"Distance change: {abs(serial_communicator.current_total_distance - last_processed_distance):.2f}mm"
            #     )

            time.sleep(0.005)

        except Exception as e:
            print(f"[ERROR] Serial monitor thread: {e}")
            shutdown_event.set()


def main():
    """Main function to start the system"""

    # Create timestamped output folder for this session
    session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_output_dir = os.path.join(config.OUTPUT_DIR, session_timestamp)
    os.makedirs(session_output_dir, exist_ok=True)
    print(f"📁 Session output directory: {session_output_dir}")

    # Initialize MQTT heartbeat
    heartbeat = None
    try:
        heartbeat = MqttHeartbeat(
            broker=config.MQTT_SERVER,
            port=config.MQTT_PORT,
            username=config.MQTT_USERNAME,
            password=config.MQTT_PASSWORD,
            topic=config.MQTT_HEARTBEAT_TOPIC,
            interval_sec=config.MQTT_HEARTBEAT_INTERVAL,
            tls_insecure=config.MQTT_TLS_INSECURE,
        )
        heartbeat.start()
        print(f"✅ MQTT heartbeat started: {config.MQTT_HEARTBEAT_TOPIC} (every {config.MQTT_HEARTBEAT_INTERVAL}s)")
    except Exception as e:
        print(f"⚠️ MQTT heartbeat not started: {e} (continuing without heartbeat)")

    print("🚀 STARTING OPTIMIZED FABRIC INSPECTION SYSTEM")
    print("=" * 50)
    print("System Architecture:")
    print("  • Esp32 with rotary encoder sends stitch count to PC via serial")
    print("  • MySQL: Insert ON EVERY processed frame in 2 seconds ")
    print("  • Image Cleanup: Deletes images older than 24 hours")
    print("=" * 50)
    print(f"Data will be inserted into MySQL at {config.DB_CONFIG['host']}/{config.DB_CONFIG['database']}")
    print(f"Images in {config.OUTPUT_DIR} will be deleted after {config.IMAGE_RETENTION_SECONDS/3600:.1f} hours")
    print("=" * 50)

    # Initialize components
    print("🤖 Loading AI model...")
    model = YOLO("best_curve_100.pt")
    model.to(config.DEVICE)
    print(f"✅ Model loaded on {config.DEVICE}")

    camera_manager = CameraManager()
    if not camera_manager.cap:
        print("❌ CRITICAL ERROR: Camera initialization failed")
        sys.exit(1)

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
            stitch_length_buffer.append(stitch_val)
            seam_allowance_buffer.append(seam_val)
        if recent_real:
            print(
                f"🔄 Seeded measurement buffers from DB: {len(recent_real)} samples "
                f"(stitch mean {sum(stitch_length_buffer)/len(stitch_length_buffer):.3f}mm, "
                f"seam mean {sum(seam_allowance_buffer)/len(seam_allowance_buffer):.3f}mm)"
            )



    threads = []

    if serial_communicator.serial_port is not None:
        serial_thread = threading.Thread(
            target=serial_monitor_thread,
            args=(serial_communicator, image_processor, camera_manager, db_manager, session_output_dir),
            daemon=True
        )
        serial_thread.start()
        threads.append(serial_thread)
        print("✅ Serial monitor thread started")
    else:
        print("⚠️ Serial monitor thread not started: Serial port not available.")

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