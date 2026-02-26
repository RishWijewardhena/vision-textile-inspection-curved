
import os
import time
import config

def image_cleanup_thread(shutdown_event):
    """Thread that deletes images older than IMAGE_RETENTION_SECONDS"""
    print("[INFO] Image cleanup thread started")
    while not shutdown_event.is_set():
        try:
            current_time = time.time()
            for filename in os.listdir(config.OUTPUT_DIR):
                if filename.endswith('.jpg'):
                    file_path = os.path.join(config.OUTPUT_DIR, filename)
                    try:
                        file_creation_time = os.path.getctime(file_path)
                        file_age = current_time - file_creation_time
                        if file_age > config.IMAGE_RETENTION_SECONDS:
                            os.remove(file_path)
                            print(f"🗑️ Deleted old image: {file_path} (Age: {file_age:.0f}s)")
                    except Exception as e:
                        print(f"[ERROR] Failed to delete {file_path}: {e}")
            time.sleep(config.CLEANUP_INTERVAL)
        except Exception as e:
            print(f"[ERROR] Image cleanup thread: {e}")
            time.sleep(config.CLEANUP_INTERVAL)
    print("[INFO] Image cleanup thread shutting down")
