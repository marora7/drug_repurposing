"""
Transform Edges Data Script

This script connects to the SQLite database, groups edge data according to provided SQL commands,
and then exports the transformed (grouped) edges data to a compressed CSV file. All configuration
is loaded from config/config.yaml.
"""

import os
import sqlite3
import logging
import argparse

from utils.config_utils import load_config
from utils.export_utils import export_table_to_csv

def group_edges(cursor, grouping_config):
    """
    Groups edge data by executing SQL statements provided in the configuration.
    It first drops any existing grouped table and then creates a new grouped table.
    """
    try:
        logging.info("Dropping existing 'edges_grouped' table if it exists.")
        cursor.execute(grouping_config["drop_table"])
        logging.info("Creating 'edges_grouped' table.")
        cursor.execute(grouping_config["create_table"])
        logging.info("Grouped table 'edges_grouped' created successfully.")

        # Optionally, log sample data from the grouped table.
        cursor.execute("SELECT * FROM edges_grouped LIMIT 10;")
        sample = cursor.fetchall()
        headers = [desc[0] for desc in cursor.description]
        logging.info("Sample data from 'edges_grouped': %s", headers)
        for row in sample:
            logging.info(row)
    except Exception as e:
        logging.error("Error during grouping edges: %s", e)
        raise

def main():
    parser = argparse.ArgumentParser(
        description="Transform edges data and export to CSV."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML configuration file (e.g., config/config.yaml)"
    )
    args = parser.parse_args()

    # Load configuration from YAML file.
    config = load_config(args.config)
    db_path = config["database"]["pubtator_db"]
    edges_config = config["transformation"]["edges"]
    grouping_config = edges_config["grouping"]
    export_query = edges_config["export_query"]
    output_path = edges_config["output"]

    if not os.path.exists(db_path):
        logging.error("Database file not found at %s", db_path)
        raise FileNotFoundError(f"Database file not found: {db_path}")

    logging.info("Connecting to database at %s", db_path)
    connection = sqlite3.connect(db_path)
    try:
        cursor = connection.cursor()
        # First, perform the grouping operation.
        group_edges(cursor, grouping_config)

        # For edges, we want to explicitly define headers (since our query does not provide them).
        headers = [":START_ID", "node1_type", ":END_ID", "node2_type", ":TYPE", "pmcount:int"]

        # Use the generalized export function for the grouped edges.
        export_table_to_csv(
            connection=connection,
            query=export_query,
            output_path=output_path,
            headers=headers,   # Provide headers explicitly for edges
            row_transform=None # No additional row transformation needed
        )
        connection.commit()
        logging.info("Database changes committed successfully.")
    except Exception as e:
        logging.error("Error during edges export: %s", e)
        raise
    finally:
        connection.close()
        logging.info("Database connection closed.")

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, 
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    main()
