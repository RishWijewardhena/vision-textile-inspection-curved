# database_manager.py

import mysql.connector
from mysql.connector import Error
from datetime import datetime
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

    def insert_measurement(self, stitch_length, seam_allowance, total_distance) -> bool:
        """
        Insert a measurement record.

        Behavior:
          - Real-data only mode: skip insert when any required value is missing.
        """
        if not self.connect():
            return False

        if stitch_length is None or seam_allowance is None or total_distance is None:
            print(
                "⚠️ Skipping DB insert: real measurement missing "
                f"(stitch_length={stitch_length}, seam_allowance={seam_allowance}, total_distance={total_distance})"
            )
            return False

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

        reset_query = f"""
        INSERT INTO `{self.db_table}`
        (`timestamp`, `stitch_length`, `seam_allowance`, `total_distance`)
        VALUES (NOW(), 0, 0, 0)
        """
        try:
            self.cursor.execute(reset_query)
            self.connection.commit()
            print("✅ Total distance reset to 0")
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

    def get_last_total_distance(self):
        """Get the total_distance of the last measurement in the database."""
        if not self.connect():
            return None

        query = f"SELECT `total_distance` FROM `{self.db_table}` ORDER BY `timestamp` DESC LIMIT 1"
        try:
            self.cursor.execute(query)
            result = self.cursor.fetchone()
            if result:
                return float(result[0])
            else:
                return 0.0
        except Error as e:
            print(f"❌ Failed to fetch last total distance: {e}")
            return None

    def get_recent_valid_measurements(self, limit=5):
        """Get recent non-null positive stitch and seam values for fallback buffers."""
        if not self.connect():
            return []

        query = f"""
        SELECT `stitch_length`, `seam_allowance`
        FROM `{self.db_table}`
        WHERE `stitch_length` IS NOT NULL
          AND `seam_allowance` IS NOT NULL
          AND `stitch_length` > 0
          AND `seam_allowance` > 0
        ORDER BY `timestamp` DESC
        LIMIT %s
        """

        try:
            self.cursor.execute(query, (int(limit),))
            rows = self.cursor.fetchall() or []
            rows.reverse()  # Return oldest -> newest for smoother buffer stats
            return [(float(r[0]), float(r[1])) for r in rows]
        except Error as e:
            print(f"❌ Failed to fetch recent valid measurements: {e}")
            return []


if __name__ == "__main__":
    # Example usage
    with DatabaseManager() as db:
        db.reset_total_distance_on_startup()
        last_date = db.get_last_measurement_date()
        print(f"Last measurement date: {last_date}")
        last_total_distance = db.get_last_total_distance()
        print(f"Last total distance: {last_total_distance}")