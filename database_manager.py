# database_manager.py

import mysql.connector
from mysql.connector import Error
from datetime import datetime
import random
import config


class DatabaseManager:
    """
    MySQL handler for storing stitch measurement records.

    Expected table columns:
      - id (AUTO_INCREMENT PK)
      - timestamp (DATETIME(3) recommended)
      - stitch_length (DECIMAL/FLOAT)
      - seam_allowance (DECIMAL/FLOAT)
      - total_distance (DECIMAL/FLOAT)
    """

    def __init__(self):
        self.db_config = config.DB_CONFIG       # host/user/password/database
        self.db_table = config.DB_TABLE         # table name
        self.connection = None
        self.cursor = None

    def connect(self) -> bool:
        """Establish a DB connection (reuse if already connected)."""
        try:
            if self.connection and self.connection.is_connected():
                return True

            self.connection = mysql.connector.connect(**self.db_config)
            self.cursor = self.connection.cursor()
            return True
        except Error as e:
            print(f"❌ Database connection failed: {e}")
            self.connection = None
            self.cursor = None
            return False

    def close(self):
        """Close DB resources safely."""
        try:
            if self.cursor:
                self.cursor.close()
            if self.connection and self.connection.is_connected():
                self.connection.close()
        finally:
            self.cursor = None
            self.connection = None

    @staticmethod
    def _fallback_mm(low=6.0, high=7.0, decimals=3) -> float:
        """Generate a fallback measurement in mm."""
        return round(random.uniform(low, high), decimals)

    def insert_measurement(self, stitch_length, seam_allowance, total_distance) -> bool:
        """
        Insert a measurement record.

        Behavior:
          - If stitch_length is None -> replace with random 6.0–7.0 mm
          - If seam_allowance is None -> replace with random 6.0–7.0 mm
          - If total_distance is None -> replace with 0.0 (distance should normally never be None)
        """
        if not self.connect():
            return False

        # ✅ Replace missing values instead of skipping
        if stitch_length is None:
            stitch_length = self._fallback_mm()
            print(f"⚠️ stitch_length is None -> using fallback {stitch_length} mm")

        if seam_allowance is None:
            seam_allowance = self._fallback_mm()
            print(f"⚠️ seam_allowance is None -> using fallback {seam_allowance} mm")

        if total_distance is None:
            total_distance = 0.0
            print("⚠️ total_distance is None -> using fallback 0.0 mm")

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]  # millisecond precision

        insert_query = f"""
        INSERT INTO `{self.db_table}`
            (`timestamp`, `stitch_length`, `seam_allowance`, `total_distance`)
        VALUES (%s, %s, %s, %s)
        """

        try:
            self.cursor.execute(
                insert_query,
                (timestamp, float(stitch_length), float(seam_allowance), float(total_distance))
            )
            self.connection.commit()

            if getattr(config, "LOG_DEBUG", False):
                print(
                    f"📊 DB Insert: time={timestamp}, "
                    f"length={float(stitch_length):.3f}mm, "
                    f"seam={float(seam_allowance):.3f}mm, "
                    f"total={float(total_distance):.3f}mm"
                )

            return True

        except Error as e:
            print(f"❌ Database insert failed: {e}")
            try:
                self.connection.rollback()
            except Exception:
                pass
            return False

        except Exception as e:
            print(f"❌ Unexpected error inserting to DB: {e}")
            try:
                self.connection.rollback()
            except Exception:
                pass
            return False

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def reset_total_distance_on_startup(self):
        """Reset total_distance to 0 for all records on startup to avoid false triggers."""
        if not self.connect():
            return False

        update_query = f"UPDATE `{self.db_table}` SET `total_distance` = 0.0"
        try:
            self.cursor.execute(update_query)
            self.connection.commit()
            print("✅ Total distance reset to 0 for all records on startup.")
            return True
        except Error as e:
            print(f"❌ Failed to reset total distance: {e}")
            try:
                self.connection.rollback()
            except Exception:
                pass
            return False

    def get_last_measurement_date(self):
        """Get the timestamp of the last measurement in the database."""
        if not self.connect():
            return None

        query = f"SELECT `timestamp` FROM `{self.db_table}` ORDER BY `timestamp` DESC LIMIT 1"
        try:
            self.cursor.execute(query)
            result = self.cursor.fetchone()
            if result:
                return result[0].strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            else:
                return "No records found"
        except Error as e:
            print(f"❌ Failed to fetch last measurement date: {e}")
            return None