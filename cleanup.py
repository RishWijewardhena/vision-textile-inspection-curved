
import os
import time
import config


def _is_inside_active_session(path, active_session_dir):
    if not active_session_dir:
        return False
    abs_path = os.path.abspath(path)
    return abs_path == active_session_dir or abs_path.startswith(active_session_dir + os.sep)

def image_cleanup_thread(shutdown_event, active_session_dir=None):
    """Thread that deletes images and folders older than IMAGE_RETENTION_SECONDS"""
    print("[INFO] Image cleanup thread started")
    active_session_dir = os.path.abspath(active_session_dir) if active_session_dir else None
    while not shutdown_event.is_set():
        try:
            current_time = time.time()
            deleted_any = False

            # Delete old .jpg files recursively (including files directly under OUTPUT_DIR).
            for root, dirs, files in os.walk(config.OUTPUT_DIR, topdown=True):
                if _is_inside_active_session(root, active_session_dir):
                    dirs[:] = []
                    continue

                # Skip walking into the active session subtree.
                dirs[:] = [
                    d for d in dirs
                    if not _is_inside_active_session(os.path.join(root, d), active_session_dir)
                ]

                for filename in files:
                    if not filename.lower().endswith('.jpg'):
                        continue

                    file_path = os.path.join(root, filename)
                    try:
                        file_age = current_time - os.path.getmtime(file_path)
                        if file_age > config.IMAGE_RETENTION_SECONDS:
                            os.remove(file_path)
                            #print(f"🗑️ Deleted old image: {file_path}")
                            deleted_any = True
                    except Exception as e:
                        print(f"[ERROR] Failed to process file {file_path}: {e}")

            # Remove empty folders (bottom-up), excluding OUTPUT_DIR and active session dir.
            for root, dirs, _ in os.walk(config.OUTPUT_DIR, topdown=False):
                for d in dirs:
                    folder_path = os.path.join(root, d)
                    if _is_inside_active_session(folder_path, active_session_dir):
                        continue
                    try:
                        if os.path.isdir(folder_path) and not os.listdir(folder_path):
                            os.rmdir(folder_path)
                            print(f"🗑️ Removed empty folder: {folder_path}")
                            deleted_any = True
                    except Exception as e:
                        print(f"[ERROR] Failed to remove empty folder {folder_path}: {e}")

            if not deleted_any:
                print("[INFO] Cleanup check complete - no old folders found")

            time.sleep(config.CLEANUP_INTERVAL)
        except Exception as e:
            print(f"[ERROR] Image cleanup thread: {e}")
            time.sleep(config.CLEANUP_INTERVAL)
    print("[INFO] Image cleanup thread shutting down")
