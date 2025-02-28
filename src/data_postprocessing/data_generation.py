"""
Data generation script for creating relations tables in the database.
This script creates two tables:
1. relations_before_prediction - combines data from nodes and edges CSV files
2. relations_after_prediction - transforms data from treat_relations and relations_before_prediction
"""
import os
import sys
import sqlite3
import pandas as pd
import gzip
import logging
import argparse
from src.utils.config_utils import load_config

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_relations_before_prediction_table(config_path):
    """
    Creates a new table 'relations_before_prediction' in the existing database
    by combining data from nodes.csv.gz and edges.csv.gz files.
    
    Args:
        config_path (str): Path to the configuration file
    """
    try:
        # Load configuration
        config = load_config(config_path)
        
        # Get paths from config
        db_path = config['ml_model']['db_path']
        nodes_path = config['nodes_file']
        edges_path = config['edges_file']
        
        # Get SQL queries from config
        create_table_query = config['post_processing']['create_relations_before_prediction']
        insert_data_query = config['post_processing']['insert_relations_before_prediction']
        
        logger.info("Loading nodes data...")
        # Load nodes data
        with gzip.open(nodes_path, 'rt') as f:
            nodes_df = pd.read_csv(f, sep=',')
        
        # Create a dictionary mapping node IDs to their names for quick lookup
        logger.info("Creating node ID to name mapping...")
        node_id_to_name = dict(zip(nodes_df['Id:ID'], nodes_df['name']))
        
        logger.info("Loading edges data...")
        # Load edges data
        with gzip.open(edges_path, 'rt') as f:
            edges_df = pd.read_csv(f, sep=',')
        
        # Connect to the database
        logger.info(f"Connecting to database at {db_path}...")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create the new table
        logger.info("Creating relations_before_prediction table...")
        cursor.execute(create_table_query)
        
        # Process and insert the data in batches to avoid memory issues
        batch_size = 10000
        total_rows = len(edges_df)
        
        for i in range(0, total_rows, batch_size):
            logger.info(f"Processing batch {i//batch_size + 1}/{(total_rows//batch_size) + 1}...")
            batch = edges_df.iloc[i:i+batch_size]
            
            # Prepare the data for insertion
            data_to_insert = []
            for _, row in batch.iterrows():
                start_id = row[':START_ID']
                end_id = row[':END_ID']
                
                # Get the names from the mapping
                start_id_name = node_id_to_name.get(start_id, "Unknown")
                end_id_name = node_id_to_name.get(end_id, "Unknown")
                
                data_to_insert.append((
                    start_id,
                    row['node1_type'],
                    start_id_name,
                    end_id,
                    row['node2_type'],
                    end_id_name,
                    row[':TYPE'],
                    row['pmcount:int']
                ))
            
            # Insert the batch
            cursor.executemany(insert_data_query, data_to_insert)
            conn.commit()
        
        # Create indexes for better query performance
        logger.info("Creating indexes on the new table...")
        for index_query in config['post_processing'].get('indexes_relations_before_prediction', []):
            cursor.execute(index_query)
            conn.commit()
        
        logger.info("Table creation and data insertion completed successfully!")
        conn.close()
        
    except Exception as e:
        logger.error(f"Error creating relations_before_prediction table: {str(e)}")
        raise

def create_relations_after_prediction_table(config_path):
    """
    Creates a new table 'relations_after_prediction' in the existing database
    based on data from 'treat_relations' and 'relations_before_prediction' tables.
    
    Args:
        config_path (str): Path to the configuration file
    """
    try:
        # Load configuration
        config = load_config(config_path)
        
        # Get database path from config
        db_path = config['ml_model']['db_path']
        
        # Get SQL queries from config
        create_table_query = config['post_processing']['create_relations_after_prediction']
        insert_data_query = config['post_processing']['insert_relations_after_prediction']
        
        # Connect to the database
        logger.info(f"Connecting to database at {db_path}...")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create the new table
        logger.info("Creating relations_after_prediction table...")
        cursor.execute(create_table_query)
        conn.commit()
        
        # Execute the insert query that pulls and transforms data from treat_relations and relations_before_prediction
        logger.info("Inserting data into relations_after_prediction table...")
        cursor.execute(insert_data_query)
        conn.commit()
        
        # Create indexes for better query performance
        logger.info("Creating indexes on the new table...")
        for index_query in config['post_processing'].get('indexes_relations_after_prediction', []):
            cursor.execute(index_query)
            conn.commit()
        
        # Log the count of records inserted
        cursor.execute("SELECT COUNT(*) FROM relations_after_prediction")
        count = cursor.fetchone()[0]
        logger.info(f"Successfully inserted {count} records into relations_after_prediction table")
        
        conn.close()
        logger.info("Table creation and data insertion completed successfully!")
        
    except Exception as e:
        logger.error(f"Error creating relations_after_prediction table: {str(e)}")
        raise

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Create relations tables')
    parser.add_argument('--config_path', type=str, default='src/config/config.yaml',
                        help='Path to configuration file')
    return parser.parse_args()

def main():
    """Main function to execute the script."""
    # Parse command line arguments
    args = parse_args()
    
    # Load config
    config_path = args.config_path
    
    # Create the relations_before_prediction table
    logger.info("Creating relations_before_prediction table...")
    create_relations_before_prediction_table(config_path)
    
    # Create the relations_after_prediction table
    logger.info("Creating relations_after_prediction table...")
    create_relations_after_prediction_table(config_path)

if __name__ == "__main__":
    main()