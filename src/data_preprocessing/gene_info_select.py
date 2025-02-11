"""
Preprocessing for filtering human gene nodes.

This script deletes rows from the 'nodes' table where node_type is 'Gene'
and where the node_id does not appear in the 'homo_sapiens_genes' table.
"""

import os
import sqlite3
import logging
import yaml

from src.utils.file_utils import log_progress

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

def filter_human_genes(db_path, ncbi_db_path, nodes_table, genes_table, gene_index_column, queries):
    """
    Deletes rows from the nodes table where node_type is 'Gene' and the node_id 
    does not exist in the genes table.
    
    Args:
        db_path (str): Path to the SQLite database.
        nodes_table (str): Name of the nodes table.
        genes_table (str): Name of the genes table (e.g., 'homo_sapiens_genes').
        gene_index_column (str): The column used for matching (e.g., "GeneID").
        queries (dict): A dictionary of SQL query templates for gene_info_select.
    """
    try:
        connection = sqlite3.connect(db_path)
        cursor = connection.cursor()

        # Attach the ncbi.db database with the alias 'ncbi'
        cursor.execute("ATTACH DATABASE ? AS ncbi", (ncbi_db_path,))

        # Display initial row counts by node_type.
        initial_counts_query = queries['nodes_counts_query'].format(nodes_table=nodes_table)
        cursor.execute(initial_counts_query)
        initial_counts = cursor.fetchall()
        log_progress("Initial row counts by node_type:")
        for row in initial_counts:
            log_progress(str(row))

        # Delete rows from nodes where node_type is 'Gene'
        # and node_id is not present in the genes table in the attached ncbi db.
        delete_query = queries['delete_query'].format(
            nodes_table=nodes_table,
            genes_table=genes_table,
            gene_index_column=gene_index_column
        )
        cursor.execute(delete_query)
        connection.commit()
        log_progress(f"Deleted {cursor.rowcount} rows from '{nodes_table}' where node_type = 'Gene' and no matching {gene_index_column} was found.")

        # Display final row counts by node_type.
        final_counts_query = queries['nodes_counts_query'].format(nodes_table=nodes_table)
        cursor.execute(final_counts_query)
        final_counts = cursor.fetchall()
        log_progress("Final row counts by node_type:")
        for row in final_counts:
            log_progress(str(row))
    except sqlite3.Error as e:
        logging.error(f"Error occurred during filtering human genes: {e}")
        raise
    finally:
        if 'connection' in locals() and connection:
            connection.close()
            log_progress("Database connection closed.")

def main():
    config = load_config()
    # Use pubtator.db because the nodes table is there.
    db_path = config['database']['pubtator_db']
    # Retrieve ncbi_db path for attaching.
    ncbi_db_path = config['database']['ncbi_db']
    nodes_table = config['nodes']['table_name']       # e.g., "nodes"
    genes_table = config['genes']['table_name']         # e.g., "homo_sapiens_genes"
    gene_index_column = config['genes']['index_column'] # e.g., "GeneID"

    # Get the SQL query templates from the gene_info_select section.
    queries = config['gene_info_select']

    filter_human_genes(db_path, ncbi_db_path, nodes_table, genes_table, gene_index_column, queries)

if __name__ == "__main__":
    main()
