
import os
import time
import config
from datetime import datetime

def image_cleanup_thread(shutdown_event, active_session_dir=None):
    """Thread that deletes images and folders older than IMAGE_RETENTION_SECONDS"""
    print("[INFO] Image cleanup thread started")
    active_session_dir = os.path.abspath(active_session_dir) if active_session_dir else None
    while not shutdown_event.is_set():
        try:
            current_time = time.time()
            deleted_any = False

            # Iterate through all session folders in output directory
            for folder_name in os.listdir(config.OUTPUT_DIR):
                folder_path = os.path.join(config.OUTPUT_DIR, folder_name)

                # Never delete the currently active session folder.
                if active_session_dir and os.path.abspath(folder_path) == active_session_dir:
                    continue

                # Skip if not a directory
                if not os.path.isdir(folder_path):
                    continue

                try:
                    # Get folder creation time
                    folder_creation_time = os.path.getctime(folder_path)
                    folder_age = current_time - folder_creation_time

                    if folder_age > config.IMAGE_RETENTION_SECONDS:
                        # Delete all images in folder
                        for filename in os.listdir(folder_path):
                            if filename.endswith('.jpg'):
                                file_path = os.path.join(folder_path, filename)
                                os.remove(file_path)
                                print(f"🗑️ Deleted old image: {file_path}")

                        # Remove folder if empty
                        if not os.listdir(folder_path):
                            os.rmdir(folder_path)
                            print(f"🗑️ Deleted old session folder: {folder_path} (Age: {folder_age:.0f}s)")
                            deleted_any = True
                        else:
                            print(f"⚠️ Folder not empty, skipping: {folder_path}")

                except Exception as e:
                    print(f"[ERROR] Failed to process folder {folder_path}: {e}")

            # Clean up empty folders (even if not old enough)
            for folder_name in os.listdir(config.OUTPUT_DIR):
                folder_path = os.path.join(config.OUTPUT_DIR, folder_name)
                if active_session_dir and os.path.abspath(folder_path) == active_session_dir:
                    continue
                if os.path.isdir(folder_path) and not os.listdir(folder_path):
                    try:
                        os.rmdir(folder_path)
                        print(f"🗑️ Removed empty folder: {folder_path}")
                    except Exception as e:
                        print(f"[ERROR] Failed to remove empty folder {folder_path}: {e}")

            if not deleted_any:
                print("[INFO] Cleanup check complete - no old folders found")

            time.sleep(config.CLEANUP_INTERVAL)
        except Exception as e:
            print(f"[ERROR] Image cleanup thread: {e}")
            time.sleep(config.CLEANUP_INTERVAL)
    print("[INFO] Image cleanup thread shutting down")
