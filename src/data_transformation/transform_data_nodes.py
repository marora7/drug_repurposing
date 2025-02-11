"""
Transform Nodes Data Script

This script connects to the SQLite database, fetches node data using a provided SQL query,
applies a small transformation on the node name (e.g., splitting at a '|' character), and exports 
the results to a compressed CSV file. All configuration is loaded from config/config.yaml.
"""

import os
import sqlite3
import logging
import argparse

from src.utils.config_utils import load_config
from src.utils.export_utils import export_table_to_csv

def transform_node_row(row):
    """
    Transforms a node row. If the node name (assumed to be the third column) contains a '|',
    only the first part is kept.
    """
    row = list(row)
    if len(row) >= 3 and isinstance(row[2], str) and '|' in row[2]:
        row[2] = row[2].split('|', 1)[0]
    return row

def main(cli_args=None):
    parser = argparse.ArgumentParser(
        description="Transform nodes data and export to CSV."
    )
    parser.add_argument(
        "--config",
        default="src/config/config.yaml",
        help="Path to YAML configuration file (default: src/config/config.yaml)"
    )
    # Use cli_args if provided, otherwise use sys.argv
    args = parser.parse_args(cli_args)

    # Load configuration from YAML file.
    config = load_config(args.config)
    # For nodes, we assume the PubTator database is used.
    db_path = config["database"]["pubtator_db"]
    nodes_config = config["transformation"]["nodes"]
    query = nodes_config["query"]
    output_path = nodes_config["output"]

    logging.info("Connecting to database at %s", db_path)
    connection = sqlite3.connect(db_path)
    try:
        # Use the generalized export function, providing a row transform for nodes.
        export_table_to_csv(
            connection=connection,
            query=query,
            output_path=output_path,
            row_transform=transform_node_row
        )
    except Exception as e:
        logging.error("Error during nodes export: %s", e)
        raise
    finally:
        connection.close()
        logging.info("Database connection closed.")

if __name__ == "__main__":
    main()
