"""
Utility functions for parsing files into pandas DataFrames.
"""

import pandas as pd
import logging

def parse_file(file_path, file_type):
    """
    Parses a file based on the specified file type.
    
    Parameters:
        file_path (str): Path to the file to be parsed.
        file_type (str): Type of file, which will determine the column names. 
                         Accepted values: 'disease', 'chemical', 'gene', 'relation'.
    
    Returns:
        pd.DataFrame: DataFrame obtained after parsing the file.
    
    Raises:
        ValueError: If the file_type is unrecognized.
    """
    try:
        logging.info(f"Parsing data from {file_path} for file type '{file_type}'")
        
        if file_type in ['disease', 'chemical', 'gene']:
            column_names = ["entity_id", "entity_type", "entity_label", "entity_name", "source"]
        elif file_type == 'relation':
            column_names = ["id", "entity_relation", "entity1", "entity2"]
        else:
            raise ValueError(f"Unrecognized file type for parsing: {file_type}")
        
        df = pd.read_csv(file_path, sep='\t', header=None, names=column_names)
        logging.info(f"Parsing complete for {file_path}")
        return df
    except Exception as e:
        logging.error(f"Failed to parse file {file_path}: {e}")
        raise
