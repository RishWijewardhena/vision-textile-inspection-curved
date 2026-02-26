
import time
import os
import signal
import sys
import threading
from datetime import datetime

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
last_db_insert_time = 0

def sigint_handler(sig, frame):
    print('Interrupted - shutting down threads...')
    shutdown_event.set()
    time.sleep(1)
    sys.exit(0)

signal.signal(signal.SIGINT, sigint_handler)

def process_fabric_immediate(image_processor, camera_manager, serial_communicator):
    """Process fabric immediately when triggered"""
    try:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        print(f"🔍 Starting fabric analysis at {ts}")
        frame = camera_manager.capture_frame_safely()
        if frame is None:
            print("❌ Could not capture frame - skipping analysis")
            return

        print(f"✅ Frame captured, starting AI inference...")
        start_time = time.time()
        annotated, summary, defects, result = image_processor.process_frame(frame, serial_communicator.current_total_distance)
        processing_time = time.time() - start_time

        # Update the global average stitch length for the next distance calculation
        if summary.get('avg_stitch_length_mm') is not None:
            serial_communicator.last_avg_stitch_length_mm = summary['avg_stitch_length_mm']

        out_path = os.path.join(config.OUTPUT_DIR, f"fabric_{ts}.jpg")
        cv2.imwrite(out_path, annotated)

        print(f"📊 FABRIC ANALYSIS RESULTS ({summary['timestamp']}):")
        print(f"   ├─ Total Distance: {summary['total_distance_mm']:.2f}mm")
        print(f"   ├─ Total Edges: {summary['edge_count']}")
        if summary.get('avg_stitch_length_mm') is not None:
            length_status = "❌ DEFECT" if defects.get('stitch_length', False) else "✅ OK"
            print(f"   ├─ Avg Stitch Length: {summary['avg_stitch_length_mm']:.2f}mm {length_status}")
            print(f"   ├─ Stitches per inch: {summary['stitches_per_inch']:.1f}")
        else:
            print(f"   ├─ Stitch Length: Not measurable")
        if summary.get('avg_distance_mm') is not None:
            dist_status = "❌ DEFECT" if defects.get('stitch_edge_distance', False) else "✅ OK"
            print(f"   ├─ Avg Stitch-Top Edge Distance: {summary['avg_distance_mm']:.2f}mm {dist_status}")
        else:
            print(f"   ├─ Avg Stitch-Top Edge Distance: Not measurable")
        print(f"   └─ Processing Time: {processing_time:.2f}s")

        defects_found = image_processor.process_defects((annotated, summary, defects, result), ts)
        if defects_found:
            print(f"📩 Defects detected - Data will be logged to MySQL")
        else:
            print(f"✅ NO DEFECTS - Fabric passed inspection")
        print(f"⚡ ANALYSIS COMPLETE: {processing_time:.2f}s total")

    except Exception as e:
        print(f"❌ ERROR in fabric processing: {e}")
    finally:
        if processing_lock.locked():
            processing_lock.release()

def serial_monitor_thread(serial_communicator, image_processor, camera_manager):
    """Thread that monitors serial for distance data"""
    global last_capture_time, last_processed_distance
    print("[INFO] Serial monitor thread started, reading distance data...")
    while not shutdown_event.is_set():
        try:
            serial_communicator.read_serial_data()
            current_time = time.time()
            if (current_time - last_capture_time >= config.CAPTURE_INTERVAL and
                abs(serial_communicator.current_total_distance - last_processed_distance) >= config.MIN_DISTANCE_CHANGE_MM):
                if processing_lock.acquire(blocking=False):
                    print(f"\n=== FABRIC PROCESSING TRIGGERED (Distance: {serial_communicator.current_total_distance:.2f}mm, Change: {abs(serial_communicator.current_total_distance - last_processed_distance):.2f}mm) ===")
                    processing_thread = threading.Thread(
                        target=process_fabric_immediate,
                        args=(image_processor, camera_manager, serial_communicator),
                        daemon=True
                    )
                    processing_thread.start()
                    last_capture_time = current_time
                    last_processed_distance = serial_communicator.current_total_distance
                else:
                    print("⚠️ WARNING: Processing lock in use - skipping capture")
            else:
                print(f"⚠️ Skipping capture: Time since last capture: {current_time - last_capture_time:.2f}s, Distance change: {abs(serial_communicator.current_total_distance - last_processed_distance):.2f}mm")
            time.sleep(0.005)
        except Exception as e:
            print(f"[ERROR] Serial monitor thread: {e}")
            shutdown_event.set()

def mysql_reporting_thread(db_manager, serial_communicator, image_processor):
    """Thread that inserts data into MySQL every 2 seconds, only if a frame was recently processed"""
    global last_db_insert_time
    print("[INFO] MySQL reporting thread started")
    while not shutdown_event.is_set():
        try:
            current_time = time.time()
            if (current_time - last_db_insert_time >= config.DB_INSERT_INTERVAL and
                current_time - image_processor.last_processed_time <= config.DB_INSERT_INTERVAL):
                db_manager.insert_data(
                    serial_communicator.current_total_distance,
                    image_processor.consecutive_stitch_length_defects,
                    image_processor.consecutive_stitch_edge_defects
                )
                last_db_insert_time = current_time
            else:
                print(f"⚠️ Skipping MySQL insert: Time since last insert: {current_time - last_db_insert_time:.2f}s, "
                      f"Time since last process: {current_time - image_processor.last_processed_time:.2f}s")

            time.sleep(0.1)
        except Exception as e:
            print(f"[ERROR] MySQL reporting thread: {e}")
            time.sleep(1)
    print("[INFO] MySQL reporting thread shutting down")

def main():
    """Main function to start the system"""
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
    print("  • Arduino: Motor control + distance data")
    print("  • MySQL: Data insertion every 2 seconds")
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
    serial_communicator = SerialCommunicator()

    # Start threads
    threads = []
    if serial_communicator.serial_port is not None:
        serial_thread = threading.Thread(target=serial_monitor_thread, args=(serial_communicator, image_processor, camera_manager), daemon=True)
        serial_thread.start()
        threads.append(serial_thread)
        print("✅ Serial monitor thread started")
    else:
        print("⚠️ Serial monitor thread not started: Serial port not available.")

    mysql_thread = threading.Thread(target=mysql_reporting_thread, args=(db_manager, serial_communicator, image_processor), daemon=True)
    mysql_thread.start()
    threads.append(mysql_thread)
    print("✅ MySQL reporting thread started")

    cleanup_thread = threading.Thread(target=image_cleanup_thread, args=(shutdown_event,), daemon=True)
    cleanup_thread.start()
    threads.append(cleanup_thread)
    print("✅ Image cleanup thread started")

    print(f"🎯 System ready! Processing fabric...")
    print("-" * 50)

    try:
        while not shutdown_event.is_set():
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n🛑 Shutdown requested...")
        if heartbeat:
            heartbeat.stop()
        shutdown_event.set()

    print("🔄 Waiting for threads to finish...")
    for t in threads:
        t.join(timeout=2.0)

    camera_manager.release()
    serial_communicator.close()
    print("✅ System shutdown complete")

if __name__ == "__main__":
    main()
