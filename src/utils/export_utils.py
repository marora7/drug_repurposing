# Utility module for exporting SQL query results to a gzip-compressed CSV file.

import os
import csv
import gzip
import logging

from .file_utils import ensure_directory_exists

logger = logging.getLogger(__name__)

def export_table_to_csv(connection, query, output_path, headers=None, row_transform=None):
    """
    Executes a SQL query on the provided SQLite connection and exports the result
    to a gzip-compressed CSV file.

    Args:
        connection: SQLite connection object.
        query (str): SQL query to execute.
        output_path (str): Path where the output CSV file (gzip compressed) will be saved.
        headers (list, optional): List of column headers to write. If None, headers are derived from the query.
        row_transform (function, optional): A function that takes a row tuple as input and returns a modified row.
    """
    cursor = connection.cursor()
    logger.info("Executing query: %s", query)
    cursor.execute(query)

    # Use provided headers or derive them from the cursor description
    if headers is None:
        headers = [desc[0] for desc in cursor.description]

    # Ensure the output directory exists
    output_dir = os.path.dirname(output_path)
    ensure_directory_exists(output_dir)
    logger.info("Exporting data to %s", output_path)

    # Write results to the gzip CSV file
    with gzip.open(output_path, mode='wt', newline='', encoding='utf-8') as gz_file:
        writer = csv.writer(gz_file)
        writer.writerow(headers)
        for row in cursor:
            if row_transform:
                row = row_transform(row)
            writer.writerow(row)

    logger.info("Data exported successfully to %s", output_path)
