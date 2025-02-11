"""
Utility functions for performing SQLite database operations.
"""

import sqlite3
import logging
import pandas as pd

def setup_database(db_path, table_queries):
    """
    Sets up an SQLite database by executing a set of CREATE TABLE queries.
    
    Parameters:
        db_path (str): Path to the SQLite database file.
        table_queries (dict): A dictionary mapping table names to their corresponding
                              CREATE TABLE SQL statements.
    
    Example:
        table_queries = {
            "diseases": (
                "CREATE TABLE IF NOT EXISTS diseases ("
                "   entity_id TEXT, "
                "   entity_type TEXT, "
                "   entity_label TEXT, "
                "   entity_name TEXT, "
                "   source TEXT"
                ")"
            ),
            "relations": (
                "CREATE TABLE IF NOT EXISTS relations ("
                "   id TEXT, "
                "   entity_relation TEXT, "        
                "   entity1 TEXT, "
                "   entity2 TEXT"
                ")"
            )
        }

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