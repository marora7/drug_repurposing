"""
Extract human gene information from NCBI and load it into an SQLite database.
"""

import os
import logging
import sqlite3
import pandas as pd
import yaml

# Import utilities from the utils modules
from src.utils.download_utils import download_file, extract_gz
from src.utils.db_utils import insert_data_from_dataframe, read_first_rows
from src.utils.file_utils import ensure_directory_exists, log_progress

# Configure basic logging (if not configured elsewhere)
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')


def load_config():
    """
    Loads the configuration from config/config.yaml.
    
    Returns:
        dict: The configuration dictionary.
    """
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def create_homo_sapiens_genes_table(db_path, table_name, ncbi_config):
    """
    Creates the homo_sapiens_genes table in the specified SQLite database using the SQL
    defined in the YAML configuration.
    
    Args:
        db_path (str): Path to the SQLite database file.
        table_name (str): Name of the table to create.
        ncbi_config (dict): NCBI configuration containing the SQL template.
    """
    try:
        logging.info(f"Creating table '{table_name}' in database '{db_path}'")
        # Retrieve the SQL template from the configuration and substitute the table name.
        sql_template = ncbi_config["sql"]
        sql_query = sql_template.format(table_name=table_name)
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        # Execute the SQL script 
        cursor.executescript(sql_query)
        conn.commit()
        conn.close()
        logging.info(f"Table '{table_name}' created successfully.")
    except sqlite3.Error as e:
        logging.error(f"Failed to create table '{table_name}': {e}")
        raise


def process_gene_info_file(db_file, table_name, gene_info_file, gene_info_url, temp_dir):
    """
    Downloads, extracts, parses, and loads the gene info file into the database.
    
    Args:
        db_file (str): Path to the SQLite database file.
        table_name (str): Name of the table to insert data.
        gene_info_file (str): Filename of the gene info gzipped file.
        gene_info_url (str): Base URL for downloading the file.
        temp_dir (str): Temporary directory for file downloads and extraction.
    """
    try:
        # Step 1: Download the gzipped gene info file.
        gene_info_gz_path = download_file(gene_info_file, gene_info_url, temp_dir)
        
        # Step 2: Extract the file.
        gene_info_extracted_path = extract_gz(gene_info_gz_path, temp_dir)
        
        # Step 3: Parse the file using pandas (this file already includes headers).
        gene_info_dataframe = pd.read_csv(gene_info_extracted_path, sep='\t')
        
        # Rename columns to match the table schema (e.g., change "#tax_id" to "[#tax_id]")
        gene_info_dataframe.rename(columns={"#tax_id": "[#tax_id]"}, inplace=True)
        
        # Step 4: Insert the data into the database table.
        rows_added = insert_data_from_dataframe(db_file, table_name, gene_info_dataframe)
        log_progress(f"Successfully processed {gene_info_file}: {rows_added} rows added to {table_name}.")
    except Exception as e:
        logging.error(f"Error processing {gene_info_file}: {e}")


def main():
    try:
        # Load configuration
        config = load_config()
        
        # Get NCBI-specific settings from the config file.
        gene_info_file = config['ncbi']['gene_info_file']         # e.g., "Homo_sapiens.gene_info.gz"
        gene_info_url  = config['ncbi']['gene_info_url']           # e.g., "https://ftp.ncbi.nih.gov/gene/DATA/GENE_INFO/Mammalia/"
        table_name     = config['ncbi']['table_name']              # e.g., "homo_sapiens_genes"
        temp_dir       = config['ncbi']['temp_dir']                # e.g., "./data/raw/ncbi/temp"
        db_file        = config['database']['ncbi_db']             # e.g., "./data/processed/ncbi.db"
        
        # Ensure the temporary directory exists.
        ensure_directory_exists(temp_dir)
        
        # Create the homo_sapiens_genes table using the SQL defined in the YAML config.
        create_homo_sapiens_genes_table(db_file, table_name, config['ncbi'])
        
        # Process the gene info file.
        process_gene_info_file(db_file, table_name, gene_info_file, gene_info_url, temp_dir)
        
        # Optional: Read and display the first rows from the table.
        df_preview = read_first_rows(db_file, table_name)
        print(f"Preview of table '{table_name}':")
        print(df_preview)
        
    except Exception as e:
        logging.critical(f"Process failed: {e}")


if __name__ == "__main__":
    main()
