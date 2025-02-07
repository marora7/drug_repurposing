#!/usr/bin/env python3
"""
Data cleaning and preprocessing for edges (Knowledge Graph).

This script:
  - Drops the existing 'edges' table (if any) and then creates a new one.
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

def create_edges_table(connection, edges_table):
    """
    Drops the 'edges' table if it exists, creates a new one,
    and inserts data from the 'relations' table.
    
    Args:
        connection (sqlite3.Connection): Active database connection.
        edges_table (str): The target edges table name.
    """
    try:
        # Drop the edges table if it exists using the generalized function.
        drop_table_if_exists(connection, edges_table)
        
        # Create the edges table with the desired schema.
        create_query = f"""
            CREATE TABLE {edges_table} (
                edge_id TEXT,
                edge_type TEXT,
                node1 TEXT,
                node2 TEXT
            )
        """
        create_table(connection, create_query)
        log_progress(f"'{edges_table}' table created successfully.")

        # Insert data from the 'relations' table into the edges table.
        insert_query = f"""
            INSERT INTO {edges_table} (edge_id, edge_type, node1, node2)
            SELECT id AS edge_id, entity_relation AS edge_type, entity1 AS node1, entity2 AS node2
            FROM relations;
        """
        connection.execute(insert_query)
        connection.commit()
        log_progress("Data successfully inserted into 'edges' table from 'relations' table.")
    except sqlite3.Error as e:
        logging.error(f"Error in create_edges_table: {e}")
        raise

def create_temp_table(connection, edges_table):
    """
    Creates a temporary table with additional derived columns (splitting node fields),
    then drops the original table and renames the temporary table to the target name.
    
    Args:
        connection (sqlite3.Connection): Active database connection.
        edges_table (str): The target edges table name.
    """
    try:
        temp_query = f"""
            CREATE TABLE edges_temp AS
            SELECT *,
                   SUBSTR(node1, 1, INSTR(node1, '|') - 1) AS node1_type,
                   SUBSTR(node1, INSTR(node1, '|') + 1) AS node1_id,
                   SUBSTR(node2, 1, INSTR(node2, '|') - 1) AS node2_type,
                   SUBSTR(node2, INSTR(node2, '|') + 1) AS node2_id
            FROM {edges_table};
        """
        connection.execute(temp_query)
        connection.execute(f"DROP TABLE {edges_table};")
        connection.execute("ALTER TABLE edges_temp RENAME TO edges;")
        connection.commit()
        log_progress("Temporary table created and renamed to 'edges' successfully.")
    except sqlite3.Error as e:
        logging.error(f"Error in create_temp_table: {e}")
        raise

def display_edges_table(connection, edges_table):
    """
    Displays the row count in the edges table.
    
    Args:
        connection (sqlite3.Connection): Active database connection.
        edges_table (str): The target edges table name.
    """
    try:
        cursor = connection.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM {edges_table};")
        row_count = cursor.fetchone()[0]
        log_progress(f"Row count in '{edges_table}' table: {row_count}")
    except sqlite3.Error as e:
        logging.error(f"Error in display_edges_table: {e}")
        raise

def filter_edges_table(connection, edges_table):
    """
    Filters the edges table to keep only rows where node1_type and node2_type are 'Disease', 'Gene', or 'Chemical'.
    
    Args:
        connection (sqlite3.Connection): Active database connection.
        edges_table (str): The target edges table name.
    """
    try:
        cursor = connection.cursor()
        # Count rows before filtering.
        cursor.execute(f"SELECT COUNT(*) FROM {edges_table};")
        total_rows_before = cursor.fetchone()[0]
        log_progress(f"Total rows before filtering: {total_rows_before}")

        # Count rows that do not meet the criteria.
        cursor.execute(
            f"""
            SELECT COUNT(*)
            FROM {edges_table}
            WHERE node1_type NOT IN ('Disease', 'Gene', 'Chemical')
               OR node2_type NOT IN ('Disease', 'Gene', 'Chemical');
            """
        )
        rows_to_delete = cursor.fetchone()[0]
        log_progress(f"Number of rows to delete: {rows_to_delete}")

        # Create a filtered table.
        cursor.execute(
            f"""
            CREATE TABLE edges_filtered AS
            SELECT *
            FROM {edges_table}
            WHERE node1_type IN ('Disease', 'Gene', 'Chemical')
              AND node2_type IN ('Disease', 'Gene', 'Chemical');
            """
        )
        connection.commit()

        # Replace the original table.
        cursor.execute(f"DROP TABLE {edges_table};")
        cursor.execute("ALTER TABLE edges_filtered RENAME TO edges;")
        connection.commit()

        # Display final row count.
        cursor.execute("SELECT COUNT(*) FROM edges;")
        total_rows_after = cursor.fetchone()[0]
        log_progress(f"Final row count in 'edges' table after filtering: {total_rows_after}")
    except sqlite3.Error as e:
        logging.error(f"Error in filter_edges_table: {e}")
        raise

def find_missing_nodes(connection, edges_table):
    """
    Identifies nodes in the edges table that do not have a corresponding match in the nodes table.
    
    Args:
        connection (sqlite3.Connection): Active database connection.
        edges_table (str): The target edges table name.
    
    Returns:
        dict: Counts of missing node1, missing node2, total unique missing nodes, and rows to delete.
    """
    try:
        cursor = connection.cursor()
        # Fetch node keys from the nodes table (format: "node_type|node_id").
        cursor.execute("SELECT node_type || '|' || node_id FROM nodes;")
        nodes_set = {row[0] for row in cursor.fetchall()}

        # Fetch node1 and node2 values from the edges table.
        cursor.execute(f"SELECT node1, node2 FROM {edges_table};")
        edges = cursor.fetchall()

        missing_node1 = set()
        missing_node2 = set()
        rows_to_delete = 0

        for node1, node2 in edges:
            if node1 not in nodes_set or node2 not in nodes_set:
                rows_to_delete += 1
            if node1 not in nodes_set:
                missing_node1.add(node1)
            if node2 not in nodes_set:
                missing_node2.add(node2)

        total_missing_unique = missing_node1.union(missing_node2)
        return {
            "missing_node1_count": len(missing_node1),
            "missing_node2_count": len(missing_node2),
            "total_missing_unique_count": len(total_missing_unique),
            "rows_to_delete": rows_to_delete
        }
    except sqlite3.Error as e:
        logging.error(f"Error in find_missing_nodes: {e}")
        raise

def delete_unmatched_rows(connection, edges_table):
    """
    Deletes rows in the edges table where node1 or node2 do not have a matching entry in the nodes table.
    
    Args:
        connection (sqlite3.Connection): Active database connection.
        edges_table (str): The target edges table name.
    """
    try:
        missing_nodes = find_missing_nodes(connection, edges_table)
        log_progress(f"Count of node1 not in nodes table: {missing_nodes['missing_node1_count']}")
        log_progress(f"Count of node2 not in nodes table: {missing_nodes['missing_node2_count']}")
        log_progress(f"Total unique missing nodes: {missing_nodes['total_missing_unique_count']}")
        log_progress(f"Number of rows to delete: {missing_nodes['rows_to_delete']}")

        delete_query = f"""
            DELETE FROM {edges_table}
            WHERE node1 NOT IN (SELECT node_type || '|' || node_id FROM nodes)
               OR node2 NOT IN (SELECT node_type || '|' || node_id FROM nodes);
        """
        connection.execute(delete_query)
        connection.commit()

        cursor = connection.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM {edges_table};")
        total_rows = cursor.fetchone()[0]
        log_progress(f"New total row count of '{edges_table}' table after deleting unmatched rows: {total_rows}")
    except sqlite3.Error as e:
        logging.error(f"Error in delete_unmatched_rows: {e}")
        raise

def main():
    try:
        config = load_config()
        db_path = config['database']['pubtator_db']
        edges_table = config['edges']['table_name']  # For example, "edges"

        connection = sqlite3.connect(db_path)
        logging.info("Database connection successful.")

        # Step 1: Create the edges table from the relations table.
        create_edges_table(connection, edges_table)

        # Step 2: Create a temporary table with derived columns and rename it to 'edges'.
        create_temp_table(connection, edges_table)

        # Step 3: Display the current row count.
        display_edges_table(connection, edges_table)

        # Step 4: Filter the edges table by allowed node types.
        filter_edges_table(connection, edges_table)

        # Step 5: Delete rows in the edges table with unmatched nodes.
        delete_unmatched_rows(connection, edges_table)

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
