
import mysql.connector
from mysql.connector import Error
from datetime import datetime
import config

class DatabaseManager:
    def __init__(self):
        """
        Initializes a DatabaseManager object.

        Sets the database configuration and table name from the
        config module.

        :param self: The DatabaseManager object
        :type self: DatabaseManager
        """
        self.db_config = config.DB_CONFIG
        self.db_table = config.DB_TABLE

    def insert_data(self, total_distance, consecutive_stitch_length_defects, consecutive_stitch_edge_defects):
        """Insert defect data into MySQL database."""
        try:
            connection = mysql.connector.connect(**self.db_config)
            if connection.is_connected():
                cursor = connection.cursor()
                insert_query = f'''
                INSERT INTO {self.db_table} (time_stamp, total_distance, ng_stitch_count, ng_length)
                VALUES (%s, %s, %s, %s)
                '''
                # Determine defect status based on consecutive defect threshold
                stitch_length_defect = consecutive_stitch_length_defects >= config.CONSECUTIVE_DEFECT_THRESHOLD
                stitch_edge_defect = consecutive_stitch_edge_defects >= config.CONSECUTIVE_DEFECT_THRESHOLD
                # Prepare data
                current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                data = (
                    current_time,
                    round(total_distance, 2),
                    str(stitch_length_defect).lower(),  # Convert to 'true'/'false' string
                    str(stitch_edge_defect).lower()
                )
                # Execute query
                cursor.execute(insert_query, data)
                connection.commit()
                print(f"✅ Successfully inserted data into MySQL:")
                print(f"   Timestamp: {current_time}")
                print(f"   Total Distance: {total_distance:.2f}mm")
                print(f"   Stitch Length Defect: {'Defect' if stitch_length_defect else 'No Defect'}")
                print(f"   Stitch Edge Defect: {'Defect' if stitch_edge_defect else 'No Defect'}")
            else:
                print("❌ Failed to connect to MySQL database")
            cursor.close()
            connection.close()
            return True
        except Error as e:
            print(f"❌ MySQL error: {e}")
            return False
        except Exception as e:
            print(f"❌ Error inserting to MySQL: {e}")
            return False
