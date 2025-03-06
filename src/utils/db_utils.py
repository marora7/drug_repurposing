"""
Utility functions for performing SQLite database operations.
"""
import os
import sqlite3
import logging
import pandas as pd
from typing import Optional, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)

def setup_database(db_path, table_queries):
    """
    Sets up an SQLite database by executing a set of CREATE TABLE queries.
    
    Parameters:
        db_path (str): Path to the SQLite database file.
        table_queries (dict): A dictionary mapping table names to their corresponding
                              CREATE TABLE SQL statements.

    Returns:
        None
    """
    try:
        logging.info(f"Setting up database at {db_path}")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        for table, query in table_queries.items():
            logging.info(f"Creating table '{table}' if not exists")
            cursor.execute(query)
        conn.commit()
        conn.close()
        logging.info("Database setup complete")
    except sqlite3.Error as e:
        logging.error(f"Database setup failed: {e}")
        raise

def insert_data_from_dataframe(db_path, table_name, dataframe):
    """
    Inserts data from a pandas DataFrame into the specified table in the SQLite database.
    
    Parameters:
        db_path (str): Path to the SQLite database file.
        table_name (str): Name of the table where data will be inserted.
        dataframe (pd.DataFrame): DataFrame containing the data to be inserted.
    
    Returns:
        int: The number of rows inserted.
    """
    try:
        logging.info(f"Inserting data into table '{table_name}'")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Clear existing data in the table
        cursor.execute(f"DELETE FROM {table_name}")
        
        # Insert new data using DataFrame's to_sql method
        dataframe.to_sql(table_name, conn, if_exists='append', index=False)
        rows_inserted = len(dataframe)
        
        conn.commit()
        conn.close()
        logging.info(f"Inserted {rows_inserted} rows into table '{table_name}'")
        return rows_inserted
    except sqlite3.Error as e:
        logging.error(f"Failed to insert data into table '{table_name}': {e}")
        raise

def read_first_rows(db_path, table_name, num_rows=10):
    """
    Reads and returns the first N rows from the specified table.
    
    Parameters:
        db_path (str): Path to the SQLite database file.
        table_name (str): Name of the table to read from.
        num_rows (int): Number of rows to retrieve (default: 10).
    
    Returns:
        pd.DataFrame: DataFrame containing the first N rows of the table.
    """
    try:
        logging.info(f"Reading first {num_rows} rows from table '{table_name}'")
        conn = sqlite3.connect(db_path)
        query = f"SELECT * FROM {table_name} LIMIT {num_rows}"
        df = pd.read_sql_query(query, conn)
        conn.close()
        logging.info(f"First {num_rows} rows from '{table_name}':\n{df}")
        return df
    except sqlite3.Error as e:
        logging.error(f"Failed to read rows from table '{table_name}': {e}")
        raise

def drop_table_if_exists(connection, table_name):
    """
    Checks if a table exists in the database and drops it if found.
    
    Args:
        connection (sqlite3.Connection): Active database connection.
        table_name (str): Name of the table to drop.
    """
    try:
        cursor = connection.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?;", (table_name,))
        if cursor.fetchone():
            logging.info(f"Table '{table_name}' exists. Dropping it.")
            cursor.execute(f"DROP TABLE {table_name}")
            connection.commit()
            logging.info(f"Table '{table_name}' dropped.")
        else:
            logging.info(f"Table '{table_name}' does not exist. No drop needed.")
    except Exception as e:
        logging.error(f"Error dropping table '{table_name}': {e}")
        raise

def create_table(connection, create_query):
    """
    Creates a table using the provided SQL create statement.
    
    Args:
        connection (sqlite3.Connection): Active database connection.
        create_query (str): SQL query that creates the table.
    """
    try:
        cursor = connection.cursor()
        logging.info("Creating table using provided query.")
        cursor.executescript(create_query)
        connection.commit()
        logging.info("Table created successfully.")
    except Exception as e:
        logging.error(f"Error creating table: {e}")
        raise

def setup_sqlite_table(db_path: str, table_name: str, schema: Dict[str, str], 
                       primary_key: Optional[str] = None) -> None:
    """
    Creates a table in an existing SQLite database if it doesn't exist.
    
    Args:
        db_path (str): Path to the SQLite database file
        table_name (str): Name of the table to create
        schema (Dict[str, str]): Dictionary mapping column names to their SQLite data types
        primary_key (Optional[str]): Primary key definition, can be a column name or a composite key definition
    
    Returns:
        None
    """
    # Ensure the directory exists
    db_dir = os.path.dirname(db_path)
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir)
    
    # Construct the CREATE TABLE statement
    column_defs = [f'"{col}" {dtype}' for col, dtype in schema.items()]
    
    if primary_key:
        column_defs.append(f"PRIMARY KEY ({primary_key})")
    
    create_table_sql = f"""
    CREATE TABLE IF NOT EXISTS "{table_name}" (
        {', '.join(column_defs)}
    )
    """
    
    # Connect to the database and create the table
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(create_table_sql)
        conn.commit()
        logging.info(f"Successfully set up table '{table_name}' in database '{db_path}'")
    except sqlite3.Error as e:
        logging.error(f"SQLite error: {e}")
        raise
    finally:
        conn.close()

def init_database(db_path, config):
    """
    Initializes the database for tracking prediction progress.
    
    Parameters:
        db_path (str): Path to the SQLite database file.
        config (dict): Configuration containing SQL queries for table creation.
        
    Returns:
        None
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    # Get SQL queries from config
    create_completed_diseases_sql = config.get('sql_queries', {}).get(
        'create_completed_diseases_table',
        """
        CREATE TABLE IF NOT EXISTS completed_diseases (
            disease_id TEXT PRIMARY KEY,
            timestamp TEXT,
            status TEXT,
            error_message TEXT
        )
        """
    )
    
    create_batches_sql = config.get('sql_queries', {}).get(
        'create_batches_table',
        """
        CREATE TABLE IF NOT EXISTS batches (
            batch_id INTEGER PRIMARY KEY,
            start_time TEXT,
            end_time TEXT,
            status TEXT,
            num_diseases INTEGER
        )
        """
    )
    
    # Connect to database and create tables
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create completed_diseases table
        cursor.execute(create_completed_diseases_sql)
        
        # Create batches table
        cursor.execute(create_batches_sql)
        
        conn.commit()
        conn.close()
        logging.info(f"Database initialized at {db_path}")
    except sqlite3.Error as e:
        logging.error(f"Database initialization failed: {e}")
        raise

def get_completed_diseases(db_path, config):
    """
    Gets a list of disease IDs that have already been successfully processed.
    
    Parameters:
        db_path (str): Path to the SQLite database file.
        config (dict): Configuration containing SQL queries.
        
    Returns:
        list: List of completed disease IDs.
    """
    query = config.get('sql_queries', {}).get(
        'get_completed_diseases',
        "SELECT disease_id FROM completed_diseases WHERE status = 'completed'"
    )
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(query)
        results = cursor.fetchall()
        conn.close()
        
        # Extract disease IDs from results
        completed_diseases = [row[0] for row in results]
        logging.info(f"Found {len(completed_diseases)} completed diseases")
        return completed_diseases
    except sqlite3.Error as e:
        logging.error(f"Failed to get completed diseases: {e}")
        return []

def mark_disease_completed(db_path, disease_id, status, error_message, config):
    """
    Marks a disease as completed in the database.
    
    Parameters:
        db_path (str): Path to the SQLite database file.
        disease_id (str): ID of the disease that was processed.
        status (str): Status of the processing ('completed', 'failed', etc.)
        error_message (str): Error message if processing failed, None otherwise.
        config (dict): Configuration containing SQL queries.
        
    Returns:
        None
    """
    query = config.get('sql_queries', {}).get(
        'mark_disease_completed',
        """
        INSERT OR REPLACE INTO completed_diseases
        (disease_id, timestamp, status, error_message)
        VALUES (?, ?, ?, ?)
        """
    )
    
    try:
        timestamp = datetime.now().isoformat()
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(query, (disease_id, timestamp, status, error_message))
        conn.commit()
        conn.close()
        logging.info(f"Disease {disease_id} marked as {status}")
    except sqlite3.Error as e:
        logging.error(f"Failed to mark disease {disease_id} as {status}: {e}")

def record_batch(db_path, batch_id, num_diseases, status, end_time, config):
    """
    Records batch processing information in the database.
    
    Parameters:
        db_path (str): Path to the SQLite database file.
        batch_id (int): ID of the batch being processed.
        num_diseases (int): Number of diseases in the batch.
        status (str): Status of the batch ('started', 'completed', 'failed', etc.)
        end_time (str): End time of batch processing (ISO format), None if just starting.
        config (dict): Configuration containing SQL queries.
        
    Returns:
        None
    """
    if status == 'started':
        query = config.get('sql_queries', {}).get(
            'record_batch_start',
            """
            INSERT INTO batches
            (batch_id, start_time, status, num_diseases)
            VALUES (?, ?, ?, ?)
            """
        )
        start_time = datetime.now().isoformat()
        params = (batch_id, start_time, status, num_diseases)
    else:
        query = config.get('sql_queries', {}).get(
            'record_batch_end',
            """
            UPDATE batches
            SET end_time = ?, status = ?
            WHERE batch_id = ?
            """
        )
        end_time = end_time or datetime.now().isoformat()
        params = (end_time, status, batch_id)
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(query, params)
        conn.commit()
        conn.close()
        logging.info(f"Batch {batch_id} recorded with status {status}")
    except sqlite3.Error as e:
        logging.error(f"Failed to record batch {batch_id}: {e}")

def create_drugs_grouping(db_path, config):
    """
    Create the drugs_grouping table in the database if it doesn't exist.
    
    Args:
        db_path (str): Path to the SQLite database
        config (dict): Configuration dictionary from config.yaml
    """
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get create table query from config
        create_table_query = config.get('classification_queries', {}).get('create_drugs_grouping_table')
        
        logger.debug(f"Executing query: {create_table_query}")
        cursor.execute(create_table_query)
        conn.commit()
        logger.info("drugs_grouping table created or already exists")
    
    except Exception as e:
        logger.error(f"Error creating drugs_grouping table: {e}")
    
    finally:
        if conn:
            conn.close()

def drugs_classification_to_db(db_path, classification, config):
    """
    Save drug classification results to the database.
    
    Args:
        db_path (str): Path to the SQLite database
        classification (DrugClassification): Classification result object
        config (dict): Configuration dictionary from config.yaml
    """
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get insert query from config
        insert_query = config.get('classification_queries', {}).get('insert_drug_classification')
        
        # Extract values from classification object
        values = (
            classification.drug_id,
            classification.drug_name,
            classification.category,
            classification.category_name,
            classification.confidence,
            classification.reasoning
        )
        
        logger.debug(f"Executing query: {insert_query} with values: {values}")
        cursor.execute(insert_query, values)
        conn.commit()
        logger.info(f"Classification for {classification.drug_name} saved to database")
    
    except Exception as e:
        logger.error(f"Error saving classification to database: {e}")
    
    finally:
        if conn:
            conn.close()

def create_diseases_grouping(db_path, config):
    """
    Create the diseases_grouping table in the database if it doesn't exist.
    
    Args:
        db_path (str): Path to the SQLite database
        config (dict): Configuration dictionary from config.yaml
    """
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get create table query from config
        create_table_query = config.get('classification_queries', {}).get('create_diseases_grouping_table')
        
        logger.debug(f"Executing query: {create_table_query}")
        cursor.execute(create_table_query)
        conn.commit()
        logger.info("diseases_grouping table created or already exists")
    
    except Exception as e:
        logger.error(f"Error creating diseases_grouping table: {e}")
    
    finally:
        if conn:
            conn.close()

def diseases_classification_to_db(db_path, classification, config):
    """
    Save disease classification results to the database.
    
    Args:
        db_path (str): Path to the SQLite database
        classification (DiseaseClassification): Classification result object
        config (dict): Configuration dictionary from config.yaml
    """
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get insert query from config
        insert_query = config.get('classification_queries', {}).get('insert_disease_classification')
        
        # Extract values from classification object
        values = (
            classification.disease_id,
            classification.disease_name,
            classification.disease_type,
            classification.disease_type_name,
            classification.therapeutic_area,
            classification.therapeutic_area_name,
            classification.confidence,
            classification.reasoning
        )
        
        logger.debug(f"Executing query: {insert_query} with values: {values}")
        cursor.execute(insert_query, values)
        conn.commit()
        logger.info(f"Classification for {classification.disease_name} saved to database")
    
    except Exception as e:
        logger.error(f"Error saving classification to database: {e}")
    
    finally:
        if conn:
            conn.close()