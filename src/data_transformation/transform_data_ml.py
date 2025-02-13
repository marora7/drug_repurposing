#!/usr/bin/env python3
"""
Transform the already exported nodes and edges into ML input format.

This script reads your gzipped nodes and edges CSV files,
and transforms them into separate files for the ML pipeline:
  - For LLM input: drug.csv, disease.csv, gene.csv
  - For GNN input:
      • Association matrices: drug_disease.csv, drug_gene.csv, disease_gene.csv
      • Similarity matrices: drug_drug.csv, disease_disease.csv, gene_gene.csv
"""

import os
import argparse
import logging
import pandas as pd
import numpy as np
import yaml

from src.utils.config_utils import load_config

logger = logging.getLogger(__name__)

def transform_nodes(nodes_file, output_dir):
    """
    Read nodes CSV and export separate files for chemicals, diseases, and genes.
    
    """
    logger.info("Loading nodes from: %s", nodes_file)
    nodes = pd.read_csv(nodes_file, compression='gzip')
    nodes[":LABEL"] = nodes[":LABEL"].str.lower()

    # Filter by node type
    chemicals = nodes[nodes[":LABEL"] == "chemical"].reset_index(drop=True)
    diseases  = nodes[nodes[":LABEL"] == "disease"].reset_index(drop=True)
    genes     = nodes[nodes[":LABEL"] == "gene"].reset_index(drop=True)

    # Save files for LLM input:
    chemicals[['name']].to_csv(os.path.join(output_dir, 'drug.csv'), index=False)
    diseases[['name']].to_csv(os.path.join(output_dir, 'disease.csv'), index=False)
    genes[['name']].to_csv(os.path.join(output_dir, 'gene.csv'), index=False)

    # Create mapping dictionaries: original ID -> sequential index
    chemical_mapping = {row["Id:ID"]: idx for idx, row in chemicals.iterrows()}
    disease_mapping  = {row["Id:ID"]: idx for idx, row in diseases.iterrows()}
    gene_mapping     = {row["Id:ID"]: idx for idx, row in genes.iterrows()}

    logger.info("Nodes transformed for model input: %d chemicals, %d diseases, %d genes",
                len(chemical_mapping), len(disease_mapping), len(gene_mapping))
    return chemical_mapping, disease_mapping, gene_mapping

def transform_edges(edges_file, chemical_mapping, disease_mapping, gene_mapping, output_dir):
    """
    Read edges CSV and create association matrices:
      - drug_disease.csv: chemicals (rows) × diseases (columns)
      - drug_gene.csv: chemicals × genes
      - disease_gene.csv: diseases × genes
    """
    logger.info("Loading edges from: %s", edges_file)
    edges = pd.read_csv(edges_file, compression='gzip')
    edges['node1_type'] = edges['node1_type'].str.lower()
    edges['node2_type'] = edges['node2_type'].str.lower()

    n_chem = len(chemical_mapping)
    n_dis  = len(disease_mapping)
    n_gene = len(gene_mapping)

    drug_disease_matrix = np.zeros((n_chem, n_dis), dtype=int)
    drug_gene_matrix    = np.zeros((n_chem, n_gene), dtype=int)
    disease_gene_matrix = np.zeros((n_dis, n_gene), dtype=int)

    for _, row in edges.iterrows():
        type1 = row['node1_type']
        type2 = row['node2_type']
        id1   = row[':START_ID']
        id2   = row[':END_ID']

        if type1 == 'chemical' and type2 == 'disease':
            if id1 in chemical_mapping and id2 in disease_mapping:
                drug_disease_matrix[chemical_mapping[id1], disease_mapping[id2]] = 1
        elif type1 == 'disease' and type2 == 'chemical':
            if id2 in chemical_mapping and id1 in disease_mapping:
                drug_disease_matrix[chemical_mapping[id2], disease_mapping[id1]] = 1
        elif type1 == 'chemical' and type2 == 'gene':
            if id1 in chemical_mapping and id2 in gene_mapping:
                drug_gene_matrix[chemical_mapping[id1], gene_mapping[id2]] = 1
        elif type1 == 'gene' and type2 == 'chemical':
            if id2 in chemical_mapping and id1 in gene_mapping:
                drug_gene_matrix[chemical_mapping[id2], gene_mapping[id1]] = 1
        elif type1 == 'disease' and type2 == 'gene':
            if id1 in disease_mapping and id2 in gene_mapping:
                disease_gene_matrix[disease_mapping[id1], gene_mapping[id2]] = 1
        elif type1 == 'gene' and type2 == 'disease':
            if id2 in disease_mapping and id1 in gene_mapping:
                disease_gene_matrix[disease_mapping[id2], gene_mapping[id1]] = 1

    # Save association matrices (no header/index as expected by the ML code)
    pd.DataFrame(drug_disease_matrix).to_csv(os.path.join(output_dir, 'drug_disease.csv'),
                                             header=False, index=False)
    pd.DataFrame(drug_gene_matrix).to_csv(os.path.join(output_dir, 'drug_gene.csv'),
                                          header=False, index=False)
    pd.DataFrame(disease_gene_matrix).to_csv(os.path.join(output_dir, 'disease_gene.csv'),
                                             header=False, index=False)

    logger.info("Association matrices created: drug_disease %s, drug_gene %s, disease_gene %s",
                drug_disease_matrix.shape, drug_gene_matrix.shape, disease_gene_matrix.shape)
    return drug_disease_matrix, drug_gene_matrix, disease_gene_matrix

def compute_similarity_matrices(drug_disease_matrix, drug_gene_matrix, disease_gene_matrix, output_dir):
    """
    Compute similarity matrices:
      - drug_drug.csv: Dot product of drug_disease matrix with its transpose.
      - disease_disease.csv: Dot product of the transpose of drug_disease matrix with itself.
      - gene_gene.csv: Sum of similarity from drug_gene and disease_gene associations.
    """
    drug_drug_matrix = drug_disease_matrix.dot(drug_disease_matrix.T)
    pd.DataFrame(drug_drug_matrix).to_csv(os.path.join(output_dir, 'drug_drug.csv'),
                                          header=False, index=False)

    disease_disease_matrix = drug_disease_matrix.T.dot(drug_disease_matrix)
    pd.DataFrame(disease_disease_matrix).to_csv(os.path.join(output_dir, 'disease_disease.csv'),
                                                header=False, index=False)

    gene_similarity_from_drug = drug_gene_matrix.T.dot(drug_gene_matrix)
    gene_similarity_from_disease = disease_gene_matrix.T.dot(disease_gene_matrix)
    gene_gene_matrix = gene_similarity_from_drug + gene_similarity_from_disease
    pd.DataFrame(gene_gene_matrix).to_csv(os.path.join(output_dir, 'gene_gene.csv'),
                                          header=False, index=False)

    logger.info("Similarity matrices computed.")
    return drug_drug_matrix, disease_disease_matrix, gene_gene_matrix

def main(args):
    # Load configuration from YAML using your existing utility
    config = load_config(args.config)
    
    # Retrieve file paths from the YAML config (adjust keys based on your YAML structure)
    nodes_file = config["transformation"]["nodes"]["output"]
    edges_file = config["transformation"]["edges"]["output"]
    
    # Using ml_input_dir from YAML config 
    output_dir = config.get("ml_input_dir", "data/processed/ml_input")
    os.makedirs(output_dir, exist_ok=True)

    logger.info("Starting transformation with nodes_file: %s, edges_file: %s, output_dir: %s",
                nodes_file, edges_file, output_dir)

    # Transform nodes and create LLM input files
    chemical_mapping, disease_mapping, gene_mapping = transform_nodes(nodes_file, output_dir)
    
    # Transform edges and create association matrices for GNN input
    drug_disease_matrix, drug_gene_matrix, disease_gene_matrix = transform_edges(
        edges_file, chemical_mapping, disease_mapping, gene_mapping, output_dir
    )
    
    # Compute similarity matrices
    compute_similarity_matrices(drug_disease_matrix, drug_gene_matrix, disease_gene_matrix, output_dir)
    
    logger.info("Data transformation complete. All files saved in: %s", output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Transform nodes and edges into ML input format."
    )
    parser.add_argument("--config", type=str, default="src/config/config.yaml",
                        help="Path to YAML configuration file.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    main(args)
