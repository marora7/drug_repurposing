"""
Data cleaning and preprocessing for nodes (Knowledge Graph).

This script:
  - Drops the existing 'nodes' table (if any) and then creates a new one based on the SQL defined in the config.
  - Processes data from source tables ('diseases', 'chemicals', 'genes') into the 'nodes' table.
  - Creates indexes, displays statistics, and cleans the nodes data.
"""

import os
import sqlite3
import pandas as pd
import logging
import yaml

# Import generalized functions for table operations from the utils
from src.utils.db_utils import (
    drop_table_if_exists, 
    create_table,
    insert_data_from_dataframe, 
    read_first_rows
)
from src.utils.file_utils import ensure_directory_exists, log_progress

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


def process_table_to_nodes_with_pandas(connection, source_table, node_type_column, node_id_column, node_name_column, process_table_query):
    """
    Uses Pandas to process a source table and insert grouped data into the 'nodes' table.
    
    Args:
        connection (sqlite3.Connection): Active database connection.
        source_table (str): The source table name (e.g., 'diseases').
        node_type_column (str): Column name for node type.
        node_id_column (str): Column name for node ID.
        node_name_column (str): Column name for node name.
        process_table_query (str): SQL template for selecting data from the source table.
                                   Expected placeholders: {node_type_column}, {node_id_column}, {node_name_column}, {source_table}.
    """
    try:
        query = process_table_query.format(
            node_type_column=node_type_column,
            node_id_column=node_id_column,
            node_name_column=node_name_column,
            source_table=source_table
        )
        df = pd.read_sql_query(query, connection)
        df[node_name_column] = df[node_name_column].fillna('')
        grouped_df = df.groupby([node_type_column, node_id_column])[node_name_column].agg('|'.join).reset_index()
        grouped_df.columns = ['node_type', 'node_id', 'node_name']
        grouped_df.to_sql('nodes', connection, if_exists='append', index=False)
        log_progress(f"Data from '{source_table}' successfully processed into the 'nodes' table.")
    except sqlite3.Error as e:
        logging.error(f"Error processing table '{source_table}': {e}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error processing table '{source_table}': {e}")
        raise


def create_indexes(connection, indexes_config):
    """
    Creates indexes on the source tables and the 'nodes' table based on the provided SQL queries.
    
    Args:
        connection (sqlite3.Connection): Active database connection.
        indexes_config (dict): A dictionary of SQL index creation queries.
    """
    try:
        cursor = connection.cursor()
        logging.info("Creating indexes for source tables and nodes table...")
        for query in indexes_config.values():
            cursor.executescript(query)
        connection.commit()
        logging.info("Indexes created successfully.")
    except sqlite3.Error as e:
        logging.error(f"Error creating indexes: {e}")
        raise


def display_node_statistics(connection, stats_query, sample_query, table_name):
    """
    Displays the row count and first 5 rows for each node_type in the 'nodes' table.
    
    Args:
        connection (sqlite3.Connection): Active database connection.
        stats_query (str): SQL query template for statistics; expected to have a {table_name} placeholder.
        sample_query (str): SQL query template for sample rows; expected placeholders: {table_name}, {node_type}.
        table_name (str): The name of the nodes table.
    """
    try:
        query = stats_query.format(table_name=table_name)
        stats = pd.read_sql_query(query, connection)
        logging.info("Row counts by node_type:")
        logging.info(f"\n{stats}")

        for node_type in stats['node_type']:
            logging.info(f"First 5 rows for node_type: {node_type}")
            query_sample = sample_query.format(table_name=table_name, node_type=node_type)
            sample_data = pd.read_sql_query(query_sample, connection)
            logging.info(f"\n{sample_data}")
    except sqlite3.Error as e:
        logging.error(f"Error displaying node statistics: {e}")
        raise


def clean_nodes_table(connection, select_all_query, stats_query, table_name):
    """
    Cleans the 'nodes' table by removing rows with missing or invalid values in 'node_id' and 'node_name'.
    
    Args:
        connection (sqlite3.Connection): Active database connection.
        select_all_query (str): SQL query template to select all rows from the nodes table (expects {table_name} placeholder).
        stats_query (str): SQL query template to get row counts grouped by node_type (expects {table_name} placeholder).
        table_name (str): The name of the nodes table.
    """
    try:
        query = select_all_query.format(table_name=table_name)
        nodes_df = pd.read_sql_query(query, connection)

        missing_node_id = nodes_df['node_id'].isnull().sum() + (nodes_df['node_id'] == '').sum()
        missing_node_name = nodes_df['node_name'].isnull().sum() + (nodes_df['node_name'] == '').sum()
        logging.info(f"Found {missing_node_id} missing/empty values in 'node_id' and {missing_node_name} in 'node_name'.")

        cleaned_df = nodes_df[(nodes_df['node_id'].notnull()) & (nodes_df['node_id'] != '') &
                              (nodes_df['node_name'].notnull()) & (nodes_df['node_name'] != '')]
        logging.info(f"{missing_node_id} rows with missing 'node_id' and {missing_node_name} rows with missing 'node_name' removed.")

        invalid_node_name_rows = cleaned_df[cleaned_df['node_name'].str.fullmatch(r'\|+')]
        logging.info(f"{len(invalid_node_name_rows)} rows with 'node_name' containing only '|' characters will be removed.")
        cleaned_df = cleaned_df[~cleaned_df['node_name'].str.fullmatch(r'\|+')]

        rows_starting_with_pipe = cleaned_df[cleaned_df['node_name'].str.startswith('|')]
        logging.info(f"{len(rows_starting_with_pipe)} rows with 'node_name' starting with '|' will be updated.")
        cleaned_df.loc[:, 'node_name'] = cleaned_df['node_name'].str.lstrip('|')

        # Overwrite the nodes table with the cleaned data
        cleaned_df.to_sql(table_name, connection, if_exists='replace', index=False)

        stats = pd.read_sql_query(stats_query.format(table_name=table_name), connection)
        logging.info("Updated row counts by node_type:")
        logging.info(f"\n{stats}")

    except sqlite3.Error as e:
        logging.error(f"Error cleaning '{table_name}' table: {e}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error during cleaning of '{table_name}' table: {e}")
        raise


def main():
    try:
        # Load configuration
        config = load_config()
        db_path = config['database']['pubtator_db']
        nodes_table = config['nodes']['table_name']

        # Connect to the database
        connection = sqlite3.connect(db_path)
        logging.info("Database connection successful.")

        # Drop the 'nodes' table if it exists and create a new one using the SQL defined in config.
        drop_table_if_exists(connection, nodes_table)
        nodes_create_sql = config['nodes']['sql'].format(table_name=nodes_table)
        create_table(connection, nodes_create_sql)

        # Step 2: Create indexes on the source tables and nodes table.
        create_indexes(connection, config['indexes'])
        
        # Step 3: Process source tables into the 'nodes' table.
        logging.info("Processing source tables into the 'nodes' table using Pandas.")
        process_table_query = config['nodes']['process_table_query']
        process_table_to_nodes_with_pandas(connection, 'diseases', 'entity_type', 'entity_label', 'entity_name', process_table_query)
        process_table_to_nodes_with_pandas(connection, 'chemicals', 'entity_type', 'entity_label', 'entity_name', process_table_query)
        process_table_to_nodes_with_pandas(connection, 'genes', 'entity_type', 'entity_label', 'entity_name', process_table_query)
        connection.commit()
        logging.info("All source tables processed into the 'nodes' table successfully.")
        
        # Step 4: Display node statistics.
        display_node_statistics(connection, config['nodes']['stats_query'], config['nodes']['sample_query'], nodes_table)
        
        # Step 5: Clean the 'nodes' table.
        clean_nodes_table(connection, config['nodes']['select_all_query'], config['nodes']['stats_query'], nodes_table)
        
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
