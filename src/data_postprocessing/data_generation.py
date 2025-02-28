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

# Get the absolute path of the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the project root directory (assuming the script is in src/data_postprocessing/)
project_root = os.path.abspath(os.path.join(current_dir, '../..'))
# Add the project root to sys.path
sys.path.insert(0, project_root)

# Now import from src
from src.utils.config_utils import load_config

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
    import time
    
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
        start_time = time.time()
        # Load nodes data
        with gzip.open(nodes_path, 'rt') as f:
            nodes_df = pd.read_csv(f, sep=',')
        
        logger.info(f"Nodes data loaded in {time.time() - start_time:.2f} seconds. {len(nodes_df)} nodes found.")
        
        # Create a dictionary mapping node IDs to their names for quick lookup
        logger.info("Creating node ID to name mapping...")
        # Pre-allocate dictionary size for better performance
        node_id_to_name = dict()
        node_id_to_name = {id: name for id, name in zip(nodes_df['Id:ID'], nodes_df['name'])}
        
        # Free up memory
        del nodes_df
        
        logger.info("Loading edges data...")
        start_time = time.time()
        # Load edges data
        with gzip.open(edges_path, 'rt') as f:
            edges_df = pd.read_csv(f, sep=',')
        
        logger.info(f"Edges data loaded in {time.time() - start_time:.2f} seconds. {len(edges_df)} edges found.")
        
        # Connect to the database
        logger.info(f"Connecting to database at {db_path}...")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Optimize SQLite for performance
        logger.info("Configuring SQLite for optimal performance...")
        cursor.execute("PRAGMA journal_mode = WAL")
        cursor.execute("PRAGMA synchronous = NORMAL")
        cursor.execute("PRAGMA cache_size = -1048576")  # Use 1GB of RAM for cache
        cursor.execute("PRAGMA temp_store = MEMORY")
        cursor.execute("PRAGMA mmap_size = 30000000000")  # 30GB memory mapping
        
        # Create the new table
        logger.info("Creating relations_before_prediction table...")
        cursor.execute(create_table_query)
        
        # Process and insert the data in larger batches
        batch_size = 100000  # Increased batch size for 96GB RAM
        total_rows = len(edges_df)
        total_batches = (total_rows // batch_size) + (1 if total_rows % batch_size > 0 else 0)
        
        logger.info(f"Processing {total_rows} edges in {total_batches} batches of {batch_size}...")
        
        overall_start_time = time.time()
        
        for i in range(0, total_rows, batch_size):
            batch_start_time = time.time()
            current_batch = i // batch_size + 1
            logger.info(f"Processing batch {current_batch}/{total_batches}...")
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
            logger.info(f"Inserting batch {current_batch}...")
            cursor.executemany(insert_data_query, data_to_insert)
            conn.commit()
            
            batch_time = time.time() - batch_start_time
            records_per_second = len(batch) / batch_time
            logger.info(f"Batch {current_batch} processed in {batch_time:.2f} seconds " 
                       f"({records_per_second:.2f} records/sec)")
            
            # Estimate remaining time
            elapsed_time = time.time() - overall_start_time
            progress = current_batch / total_batches
            if progress > 0:
                estimated_total_time = elapsed_time / progress
                remaining_time = estimated_total_time - elapsed_time
                logger.info(f"Progress: {progress:.1%}. Estimated time remaining: {remaining_time/60:.1f} minutes")
        
        # Free up memory
        del edges_df
        
        # Create indexes for better query performance
        logger.info("Creating indexes on the new table...")
        for index_query in config['post_processing'].get('indexes_relations_before_prediction', []):
            index_start_time = time.time()
            logger.info(f"Creating index: {index_query[:100]}...")  # Log first 100 chars of query
            cursor.execute(index_query)
            conn.commit()
            logger.info(f"Index created in {time.time() - index_start_time:.2f} seconds")
        
        # Get final table stats
        cursor.execute("SELECT COUNT(*) FROM relations_before_prediction")
        final_count = cursor.fetchone()[0]
        total_time = time.time() - overall_start_time
        
        logger.info(f"Table creation and data insertion completed successfully!")
        logger.info(f"Total time: {total_time:.2f} seconds for {final_count} records")
        logger.info(f"Average processing speed: {final_count/total_time:.2f} records/second")
        
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
    import time
    
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
        
        # Optimize SQLite for performance
        logger.info("Configuring SQLite for optimal performance...")
        cursor.execute("PRAGMA journal_mode = WAL")
        cursor.execute("PRAGMA synchronous = NORMAL")
        cursor.execute("PRAGMA cache_size = -1048576")  # Use 1GB of RAM for cache
        cursor.execute("PRAGMA temp_store = MEMORY")
        cursor.execute("PRAGMA mmap_size = 30000000000")  # 30GB memory mapping
        
        # Create the new table
        logger.info("Creating relations_after_prediction table...")
        cursor.execute(create_table_query)
        conn.commit()
        
        # Get an estimate of how many rows will be inserted
        # This helps with progress estimation
        logger.info("Estimating number of rows to be processed...")
        try:
            cursor.execute("SELECT COUNT(*) FROM treat_relations")
            treat_relations_count = cursor.fetchone()[0]
            logger.info(f"Found approximately {treat_relations_count} rows in treat_relations")
        except sqlite3.Error as e:
            logger.warning(f"Could not get count from treat_relations: {e}")
            treat_relations_count = "unknown number of"
        
        # Execute the insert query that pulls and transforms data from treat_relations and relations_before_prediction
        logger.info(f"Inserting {treat_relations_count} rows into relations_after_prediction table...")
        logger.info("This operation could take several minutes. Please wait...")
        
        start_time = time.time()
        
        conn.execute("PRAGMA busy_timeout = 600000")  # 10 minutes timeout
        
        # Execute the insertion
        cursor.execute(insert_data_query)
        conn.commit()
        
        # Calculate processing time
        insertion_time = time.time() - start_time
        logger.info(f"Data insertion completed in {insertion_time:.2f} seconds")
        
        # Log the count of records inserted
        cursor.execute("SELECT COUNT(*) FROM relations_after_prediction")
        final_count = cursor.fetchone()[0]
        logger.info(f"Successfully inserted {final_count} records into relations_after_prediction table")
        
        # Create indexes for better query performance
        logger.info("Creating indexes on the new table...")
        index_start_time = time.time()
        
        for i, index_query in enumerate(config['post_processing'].get('indexes_relations_after_prediction', []), 1):
            index_query_start = time.time()
            logger.info(f"Creating index {i}/{len(config['post_processing'].get('indexes_relations_after_prediction', []))}: {index_query[:100]}...")
            cursor.execute(index_query)
            conn.commit()
            logger.info(f"Index created in {time.time() - index_query_start:.2f} seconds")
        
        logger.info(f"All indexes created in {time.time() - index_start_time:.2f} seconds")
        
        conn.commit()
        
        # Reset pragmas to normal for database safety
        cursor.execute("PRAGMA journal_mode = DELETE")
        cursor.execute("PRAGMA synchronous = FULL")
        conn.commit()
        
        conn.close()
        logger.info(f"Table creation and data insertion completed successfully in {time.time() - start_time:.2f} seconds!")
        
    except Exception as e:
        logger.error(f"Error creating relations_after_prediction table: {str(e)}")
        # If there's an error, try to provide more context about what happened
        if 'cursor' in locals() and 'conn' in locals():
            try:
                cursor.execute("PRAGMA journal_mode = DELETE")
                cursor.execute("PRAGMA synchronous = FULL")
                conn.close()
                logger.info("Database connection closed properly despite error")
            except Exception as close_error:
                logger.error(f"Additional error while closing database: {str(close_error)}")
        
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