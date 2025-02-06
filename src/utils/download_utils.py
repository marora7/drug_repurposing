"""
Utility functions for downloading files and extracting gzip archives.
"""

import os
import logging
import requests
import gzip

def download_file(file_name, base_url, output_dir, chunk_size=8192):
    """
    Downloads a file from the given base URL and saves it to the output directory.
    
    Parameters:
        file_name (str): Name of the file to download.
        base_url (str): Base URL where the file is hosted.
        output_dir (str): Directory where the downloaded file will be stored.
        chunk_size (int): Size of the chunk to use while downloading.
    
    Returns:
        str: The full path to the downloaded file.
    """
    url = f"{base_url}{file_name}"
    output_path = os.path.join(output_dir, file_name)
    
    if os.path.exists(output_path):
        logging.info(f"File {file_name} already exists. Skipping download.")
        return output_path

    try:
        logging.info(f"Starting download: {url}")
        with requests.get(url, stream=True) as response:
            response.raise_for_status()
            with open(output_path, 'wb') as file:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        file.write(chunk)
        logging.info(f"Downloaded {file_name} to {output_path}")
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to download {file_name}: {e}")
        raise

    return output_path

def extract_gz(input_path, output_dir):
    """
    Extracts a gzipped file into the output directory.
    
    Parameters:
        input_path (str): Full path to the .gz file.
        output_dir (str): Directory where the extracted file will be saved.
    
    Returns:
        str: The full path to the extracted file.
    """
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_path = os.path.join(output_dir, base_name)
    
    if os.path.exists(output_path):
        logging.info(f"Extracted file {output_path} already exists. Skipping extraction.")
        return output_path

    try:
        logging.info(f"Extracting {input_path} to {output_path}")
        with gzip.open(input_path, 'rb') as gz_file:
            with open(output_path, 'wb') as out_file:
                out_file.write(gz_file.read())
        logging.info(f"Extraction complete for {input_path}")
    except (OSError, gzip.BadGzipFile) as e:
        logging.error(f"Failed to extract {input_path}: {e}")
        raise

    return output_path
