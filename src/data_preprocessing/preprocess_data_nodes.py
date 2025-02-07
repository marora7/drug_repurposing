"""
Data cleaning and preprocessing for nodes (Knowledge Graph).

This script:
  - Drops the existing 'nodes' table (if any) and then creates a new one.
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


def process_table_to_nodes_with_pandas(connection, source_table, node_type_column, node_id_column, node_name_column):
    """
    Uses Pandas to process a source table and insert grouped data into the 'nodes' table.
    
    Args:
        connection (sqlite3.Connection): Active database connection.
        source_table (str): The source table name (e.g., 'diseases').
        node_type_column (str): Column name for node type.
        node_id_column (str): Column name for node ID.
        node_name_column (str): Column name for node name.
    """
    try:
        query = f"SELECT {node_type_column}, {node_id_column}, {node_name_column} FROM {source_table}"
        df = pd.read_sql_query(query, connection)
        df[node_name_column] = df[node_name_column].fillna('')
        grouped_df = df.groupby([node_type_column, node_id_column])[node_name_column].agg('|'.join).reset_index()
        grouped_df.columns = ['node_type', 'node_id', 'node_name']
        grouped_df.to_sql('nodes', connection, if_exists='append', index=False)
        log_progress(f"Data from '{source_table}' successfully processed into the 'nodes' table using Pandas.")
    except sqlite3.Error as e:
        logging.error(f"Error processing table '{source_table}' using Pandas: {e}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error processing table '{source_table}': {e}")
        raise


def create_indexes(connection):
    """
    Creates indexes on the source tables and the 'nodes' table to optimize performance.
    
    Args:
        connection (sqlite3.Connection): Active database connection.
    """
    try:
        cursor = connection.cursor()
        logging.info("Creating indexes for source tables...")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_diseases_entity ON diseases(entity_type, entity_label);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_chemicals_entity ON chemicals(entity_type, entity_label);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_genes_entity ON genes(entity_type, entity_label);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_nodes_node ON nodes(node_type, node_id);")
        connection.commit()
        logging.info("Indexes created successfully.")
    except sqlite3.Error as e:
        logging.error(f"Error creating indexes: {e}")
        raise


def display_node_statistics(connection):
    """
    Displays the row count and first 5 rows for each node_type in the 'nodes' table.
    
    Args:
        connection (sqlite3.Connection): Active database connection.
    """
    try:
        query = """
            SELECT node_type, COUNT(*) AS row_count
            FROM nodes
            GROUP BY node_type
        """
        stats = pd.read_sql_query(query, connection)
        logging.info("Row counts by node_type:")
        logging.info(f"\n{stats}")

        for node_type in stats['node_type']:
            logging.info(f"First 5 rows for node_type: {node_type}")
            query_sample = f"""
                SELECT *
                FROM nodes
                WHERE node_type = '{node_type}'
                LIMIT 5
            """
            sample_data = pd.read_sql_query(query_sample, connection)
            logging.info(f"\n{sample_data}")
    except sqlite3.Error as e:
        logging.error(f"Error displaying node statistics: {e}")
        raise


def create_composite_index_nodes(connection):
    """
    Creates a composite index 'idx_nodes' on the 'node_type' and 'node_id' columns in the 'nodes' table.
    
    Args:
        connection (sqlite3.Connection): Active database connection.
    """
    try:
        cursor = connection.cursor()
        logging.info("Creating composite index 'idx_nodes' on 'node_type' and 'node_id' in the 'nodes' table...")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_nodes ON nodes(node_type, node_id);")
        connection.commit()
        logging.info("Composite index 'idx_nodes' created successfully.")
    except sqlite3.Error as e:
        logging.error(f"Error creating composite index 'idx_nodes': {e}")
        raise


def clean_nodes_table(connection):
    """
    Cleans the 'nodes' table by removing rows with missing or invalid values in 'node_id' and 'node_name'.
    
    Args:
        connection (sqlite3.Connection): Active database connection.
    """
    try:
        query = "SELECT * FROM nodes"
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

        # Overwrite the 'nodes' table with the cleaned data
        cleaned_df.to_sql('nodes', connection, if_exists='replace', index=False)

        row_count_query = """
            SELECT node_type, COUNT(*) AS row_count
            FROM nodes
            GROUP BY node_type
        """
        stats = pd.read_sql_query(row_count_query, connection)
        logging.info("Updated row counts by node_type:")
        logging.info(f"\n{stats}")

    except sqlite3.Error as e:
        logging.error(f"Error cleaning 'nodes' table: {e}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error during cleaning of 'nodes' table: {e}")
        raise


def main():
    try:
        config = load_config()
        db_path = config['database']['pubtator_db']
        nodes_table = config['nodes']['table_name']

        # Connect to the database
        connection = sqlite3.connect(db_path)
        logging.info("Database connection successful.")

        # Directly drop the table if it exists
        drop_table_if_exists(connection, nodes_table)
        # Then create the new 'nodes' table
        create_query = f"""
            CREATE TABLE {nodes_table} (
                node_type TEXT,
                node_id TEXT,
                node_name TEXT
            )
        """
        create_table(connection, create_query)

        # Step 2: Create indexes on the source tables.
        create_indexes(connection)
        
        # Step 3: Process source tables into the 'nodes' table.
        logging.info("Processing source tables into the 'nodes' table using Pandas.")
        process_table_to_nodes_with_pandas(connection, 'diseases', 'entity_type', 'entity_label', 'entity_name')
        process_table_to_nodes_with_pandas(connection, 'chemicals', 'entity_type', 'entity_label', 'entity_name')
        process_table_to_nodes_with_pandas(connection, 'genes', 'entity_type', 'entity_label', 'entity_name')
        connection.commit()
        logging.info("All source tables processed into the 'nodes' table successfully.")
        
        # Step 4: Display node statistics.
        display_node_statistics(connection)
        
        # Step 5: Create a composite index on the 'nodes' table.
        create_composite_index_nodes(connection)
        
        # Step 6: Clean the 'nodes' table.
        clean_nodes_table(connection)
        
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
