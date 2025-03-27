"""
Main entry point for the Drug Repurposing Project pipeline.

This script allows you to run the entire pipeline or to execute a specific phase for data extraction, preprocessing and transformation phase.

Usage examples:
  - Run the entire pipeline:
      python main.py --pipeline all
  - Run only the extraction phase:
      python main.py --pipeline extraction
  - Run only the preprocessing phase:
      python main.py --pipeline preprocessing
  - Run only the transformation phase:
      python main.py --pipeline transformation
"""

import argparse
import os

# Import functions for each phase from their respective packages.
from src.data_extraction import extract_pubtator, extract_ncbi
from src.data_preprocessing import preprocess_nodes, preprocess_edges, filter_human_genes
from src.data_transformation import transform_nodes, transform_edges
from src.utils.config_utils import load_config

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

config_path = os.path.join(PROJECT_ROOT, 'config', 'config.yaml')
config = load_config(config_path)
config['database']['pubtator_db'] = os.path.join(PROJECT_ROOT, config['database']['pubtator_db'])
config['database']['ncbi_db'] = os.path.join(PROJECT_ROOT, config['database']['ncbi_db'])

def run_extraction():
    """Run the data extraction modules in the sequence."""
    print("=== Starting Data Extraction ===")
    print("Extracting data from Pubtator")
    extract_pubtator()  # Calls main() from extract_data_pubtator.py
    print("Extracting data from ncbi")
    extract_ncbi()      # Calls main() from extract_data_ncbi.py
    print("=== Data Extraction Completed ===\n")

def run_preprocessing():
    """Run the data preprocessing modules in the sequence."""
    print("=== Starting Data Preprocessing ===")
    print("Preprocessing the nodes data")
    preprocess_nodes()       # Calls main() from preprocess_data_nodes.py
    print("Preprocessing the edges data")
    preprocess_edges()       # Calls main() from preprocess_data_edges.py
    print("Filtering the human genes data")
    filter_human_genes()     # Calls main() from gene_info_select.py
    print("=== Data Preprocessing Completed ===\n")

def run_transformation():
    """Run the data transformation modules in the sequence."""
    print("=== Starting Data Transformation for Knowledge Graph===")
    print("Transforming nodes data")
    transform_nodes([])  # Pass an empty list to avoid picking up --pipeline
    print("Transforming edges data")
    transform_edges([])
    print("=== Data Transformation Completed ===\n")

def main():
    # Set up command-line argument parsing.
    parser = argparse.ArgumentParser(
        description="Drug Repurposing Pipeline Runner: Run the full pipeline or a specific phase."
    )
    parser.add_argument(
        '--pipeline',
        choices=['all', 'extraction', 'preprocessing', 'transformation'],
        default='all',
        help="Choose which pipeline phase to run: 'all' (default), 'extraction', 'preprocessing', 'transformation'."
    )

    args = parser.parse_args()

    # Execute the requested pipeline phase.
    if args.pipeline == 'all':
        run_extraction()
        run_preprocessing()
        run_transformation()
    elif args.pipeline == 'extraction':
        run_extraction()
    elif args.pipeline == 'preprocessing':
        run_preprocessing()
    elif args.pipeline == 'transformation':
        run_transformation()

if __name__ == '__main__':
    main()
