"""
Data cleaning and preprocessing for edges (Knowledge Graph).

This script:
  - Drops the existing 'edges' table (if any) and then creates a new one based on the SQL defined in the config.
  - Inserts data from the 'relations' table into the 'edges' table.
  - Creates a temporary table with derived columns (node1_type, node1_id, node2_type, node2_id) and renames it to 'edges'.
  - Filters the edges table to keep only rows with allowed node types.
  - Deletes rows with unmatched nodes (nodes not found in the 'nodes' table).
  - Displays row counts before and after filtering and deletion.
"""

import os
import sqlite3
import logging
import yaml

# Import generalized functions from our utils modules
from src.utils.db_utils import drop_table_if_exists, create_table
from src.utils.file_utils import log_progress

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def load_config():
    """
    Loads configuration from config/config.yaml.
    
    Returns:
        dict: The configuration dictionary.
    """
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_edges_table(connection, edges_table, config):
    """
    Drops the 'edges' table if it exists, creates a new one based on config,
    and inserts data from the 'relations' table.
    
    Args:
        connection (sqlite3.Connection): Active database connection.
        edges_table (str): The target edges table name.
        config (dict): The configuration dictionary.
    """
    try:
        # Drop the edges table if it exists.
        drop_table_if_exists(connection, edges_table)
        
        # Create the edges table using the SQL defined in config.
        create_sql = config['edges']['sql'].format(table_name=edges_table)
        create_table(connection, create_sql)
        log_progress(f"'{edges_table}' table created successfully.")
        
        # Insert data from the 'relations' table.
        insert_sql = config['edges']['insert_sql'].format(table_name=edges_table)
        connection.execute(insert_sql)
        connection.commit()
        log_progress("Data successfully inserted into 'edges' table from 'relations' table.")
    except sqlite3.Error as e:
        logging.error(f"Error in create_edges_table: {e}")
        raise


def create_temp_table(connection, edges_table, config):
    """
    Creates a temporary table with derived columns, then drops the original table
    and renames the temporary table to the target name.
    
    Args:
        connection (sqlite3.Connection): Active database connection.
        edges_table (str): The target edges table name.
        config (dict): The configuration dictionary.
    """
    try:
        temp_sql = config['edges']['temp_sql'].format(table_name=edges_table)
        connection.execute(temp_sql)
        # Drop the original table and rename the temporary table.
        drop_rename_sql = config['edges']['drop_temp_and_rename_sql'].format(table_name=edges_table)
        connection.executescript(drop_rename_sql)
        connection.commit()
        log_progress("Temporary table created and renamed to 'edges' successfully.")
    except sqlite3.Error as e:
        logging.error(f"Error in create_temp_table: {e}")
        raise


def display_edges_table(connection, edges_table, config):
    """
    Displays the row count in the edges table.
    
    Args:
        connection (sqlite3.Connection): Active database connection.
        edges_table (str): The target edges table name.
        config (dict): The configuration dictionary.
    """
    try:
        count_sql = config['edges']['count_sql'].format(table_name=edges_table)
        cursor = connection.cursor()
        cursor.execute(count_sql)
        row_count = cursor.fetchone()[0]
        log_progress(f"Row count in '{edges_table}' table: {row_count}")
    except sqlite3.Error as e:
        logging.error(f"Error in display_edges_table: {e}")
        raise


def filter_edges_table(connection, edges_table, config):
    """
    Filters the edges table to keep only rows with allowed node types.
    
    Args:
        connection (sqlite3.Connection): Active database connection.
        edges_table (str): The target edges table name.
        config (dict): The configuration dictionary.
    """
    try:
        cursor = connection.cursor()
        # Count rows before filtering.
        count_sql = config['edges']['count_sql'].format(table_name=edges_table)
        cursor.execute(count_sql)
        total_rows_before = cursor.fetchone()[0]
        log_progress(f"Total rows before filtering: {total_rows_before}")

        # Count rows that do not meet the criteria.
        filter_count_sql = config['edges']['filter_count_sql'].format(table_name=edges_table)
        cursor.execute(filter_count_sql)
        rows_to_delete = cursor.fetchone()[0]
        log_progress(f"Number of rows to delete (do not meet criteria): {rows_to_delete}")

        # Create a filtered table.
        filter_create_sql = config['edges']['filter_create_sql'].format(table_name=edges_table)
        cursor.executescript(filter_create_sql)
        connection.commit()

        # Replace the original table with the filtered table.
        cursor.executescript(f"DROP TABLE {edges_table};")
        cursor.executescript(f"ALTER TABLE edges_filtered RENAME TO {edges_table};")
        connection.commit()

        # Display final row count.
        cursor.execute(count_sql)
        total_rows_after = cursor.fetchone()[0]
        log_progress(f"Final row count in '{edges_table}' table after filtering: {total_rows_after}")
    except sqlite3.Error as e:
        logging.error(f"Error in filter_edges_table: {e}")
        raise


def delete_unmatched_rows(connection, edges_table, config):
    """
    Deletes rows in the edges table where node1 or node2 do not have a matching entry in the nodes table.
    
    Args:
        connection (sqlite3.Connection): Active database connection.
        edges_table (str): The target edges table name.
        config (dict): The configuration dictionary.
    """
    try:
        delete_sql = config['edges']['delete_unmatched_sql'].format(table_name=edges_table)
        connection.execute(delete_sql)
        connection.commit()

        # Display new total row count.
        count_sql = config['edges']['count_sql'].format(table_name=edges_table)
        cursor = connection.cursor()
        cursor.execute(count_sql)
        total_rows = cursor.fetchone()[0]
        log_progress(f"New total row count of '{edges_table}' table after deleting unmatched rows: {total_rows}")
    except sqlite3.Error as e:
        logging.error(f"Error in delete_unmatched_rows: {e}")
        raise


def main():
    try:
        config = load_config()
        db_path = config['database']['pubtator_db']
        edges_table = config['edges']['table_name']  # e.g., "edges"

        connection = sqlite3.connect(db_path)
        logging.info("Database connection successful.")

        # Step 1: Create the edges table from the relations table.
        create_edges_table(connection, edges_table, config)

        # Step 2: Create a temporary table with derived columns and rename it to 'edges'.
        create_temp_table(connection, edges_table, config)

        # Step 3: Display the current row count.
        display_edges_table(connection, edges_table, config)

        # Step 4: Filter the edges table by allowed node types.
        filter_edges_table(connection, edges_table, config)

        # Step 5: Delete rows in the edges table with unmatched nodes.
        delete_unmatched_rows(connection, edges_table, config)

    except sqlite3.Error as e:
        logging.error(f"Database operation failed: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
    finally:
        if 'connection' in locals() and connection:
            connection.close()
            logging.info("Database connection closed.")
        else:
            logging.error("Failed to establish database connection.")


if __name__ == "__main__":
    main()
