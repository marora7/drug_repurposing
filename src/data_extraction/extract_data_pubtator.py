#!/usr/bin/env python3
"""
Extract data from Pubtator and load it into an SQLite database.

This script downloads the Pubtator files (as configured in config/config.yaml),
extracts them, parses them into DataFrames, and inserts the data into the
respective tables in the SQLite database.
"""

import os
import logging
import yaml

# Import utility functions
from src.utils.download_utils import download_file, extract_gz
from src.utils.db_utils import setup_database, insert_data_from_dataframe, read_first_rows
from src.utils.parse_utils import parse_file
from src.utils.file_utils import ensure_directory_exists, log_progress

# Configure basic logging 
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_config():
    """
    Loads the configuration from config/config.yaml.
    
    Returns:
        dict: Configuration dictionary.
    """
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
    with open(config_path, 'r') as config_file:
        config = yaml.safe_load(config_file)
    return config


def get_table_name(file_name):
    """
    Determines the database table name based on the file name.
    
    Args:
        file_name (str): The name of the file being processed.
    
    Returns:
        str or None: The table name (e.g., 'diseases', 'chemicals', 'genes', 'relations')
                     or None if no match is found.
    """
    if "disease" in file_name:
        return "diseases"
    elif "chemical" in file_name:
        return "chemicals"
    elif "gene" in file_name:
        return "genes"
    elif "relation" in file_name:
        return "relations"
    else:
        return None


def main():
    try:
        # Load configuration
        config = load_config()
        base_url = config['pubtator']['base_url']
        files = config['pubtator']['files']
        temp_dir = config['pubtator']['temp_dir']
        chunk_size = config['download']['chunk_size']
        db_file = config['database']['pubtator_db']

        # Ensure temporary directory exists
        ensure_directory_exists(temp_dir)

        # Define SQL queries for creating tables
        table_queries = {
            "diseases": """
                CREATE TABLE IF NOT EXISTS diseases (
                    entity_id TEXT,
                    entity_type TEXT,
                    entity_label TEXT,
                    entity_name TEXT,
                    source TEXT
                )
            """,
            "chemicals": """
                CREATE TABLE IF NOT EXISTS chemicals (
                    entity_id TEXT,
                    entity_type TEXT,
                    entity_label TEXT,
                    entity_name TEXT,
                    source TEXT
                )
            """,
            "genes": """
                CREATE TABLE IF NOT EXISTS genes (
                    entity_id TEXT,
                    entity_type TEXT,
                    entity_label TEXT,
                    entity_name TEXT,
                    source TEXT
                )
            """,
            "relations": """
                CREATE TABLE IF NOT EXISTS relations (
                    id TEXT,
                    entity_relation TEXT,        
                    entity1 TEXT,
                    entity2 TEXT
                )
            """
        }

        # Setup the SQLite database with required tables
        setup_database(db_file, table_queries)

        # Initialize a dictionary to track the number of rows inserted per table
        rows_added_summary = {
            "diseases": 0,
            "chemicals": 0,
            "genes": 0,
            "relations": 0
        }

        # Process each file as specified in the configuration
        for file_name in files:
            try:
                # Step 1: Download the gzipped file
                gz_path = download_file(file_name, base_url, temp_dir, chunk_size)

                # Step 2: Extract the downloaded file
                extracted_path = extract_gz(gz_path, temp_dir)

                # Step 3: Determine the appropriate table and file type
                table_name = get_table_name(file_name)
                if table_name is None:
                    log_progress(f"Skipping unrecognized file: {file_name}")
                    continue

                # For parsing, convert table name to singular file_type
                # (e.g., "diseases" -> "disease", "relations" -> "relation")
                file_type = table_name[:-1] if table_name != "relations" else "relation"

                # Step 4: Parse the extracted file into a pandas DataFrame
                df = parse_file(extracted_path, file_type)

                # Step 5: Insert the data into the corresponding database table
                rows_inserted = insert_data_from_dataframe(db_file, table_name, df)
                rows_added_summary[table_name] += rows_inserted

            except Exception as e:
                logging.error(f"Error processing {file_name}: {e}")

        # Log a summary of the rows added to each table
        log_progress("All files processed successfully")
        for table, rows in rows_added_summary.items():
            log_progress(f"Total rows added to {table}: {rows}")

        # Optionally, display the first 10 rows from each table
        for table in rows_added_summary.keys():
            df_preview = read_first_rows(db_file, table, num_rows=10)
            print(f"\nFirst 10 rows from {table}:")
            print(df_preview)

    except Exception as e:
        logging.critical(f"Process failed: {e}")


if __name__ == "__main__":
    main()

