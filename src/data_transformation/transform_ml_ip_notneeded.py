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
      - drug_disease.csv: chemicals (rows) x diseases (columns)
      - drug_gene.csv: chemicals x genes
      - disease_gene.csv: diseases x genes
      
    This version uses chunked reading and vectorized processing to handle huge datasets.
    """
    logger.info("Loading edges from: %s", edges_file)
    
    n_chem = len(chemical_mapping)
    n_dis  = len(disease_mapping)
    n_gene = len(gene_mapping)

    drug_disease_matrix = np.zeros((n_chem, n_dis), dtype=int)
    drug_gene_matrix    = np.zeros((n_chem, n_gene), dtype=int)
    disease_gene_matrix = np.zeros((n_dis, n_gene), dtype=int)

    chunksize = 100000  # adjust this based on your memory and performance needs
    for chunk in pd.read_csv(edges_file, compression='gzip', chunksize=chunksize):
        # Convert node types to lowercase
        chunk['node1_type'] = chunk['node1_type'].str.lower()
        chunk['node2_type'] = chunk['node2_type'].str.lower()

        # Process chemical-disease edges
        mask_cd = (chunk['node1_type'] == 'chemical') & (chunk['node2_type'] == 'disease')
        cd = chunk.loc[mask_cd, [':START_ID', ':END_ID']]
        if not cd.empty:
            cd['row'] = cd[':START_ID'].map(chemical_mapping)
            cd['col'] = cd[':END_ID'].map(disease_mapping)
            cd = cd.dropna(subset=['row', 'col'])
            rows = cd['row'].astype(int).values
            cols = cd['col'].astype(int).values
            drug_disease_matrix[rows, cols] = 1

        # Process disease-chemical edges
        mask_dc = (chunk['node1_type'] == 'disease') & (chunk['node2_type'] == 'chemical')
        dc = chunk.loc[mask_dc, [':START_ID', ':END_ID']]
        if not dc.empty:
            dc['row'] = dc[':END_ID'].map(chemical_mapping)
            dc['col'] = dc[':START_ID'].map(disease_mapping)
            dc = dc.dropna(subset=['row', 'col'])
            rows = dc['row'].astype(int).values
            cols = dc['col'].astype(int).values
            drug_disease_matrix[rows, cols] = 1

        # Process chemical-gene edges
        mask_cg = (chunk['node1_type'] == 'chemical') & (chunk['node2_type'] == 'gene')
        cg = chunk.loc[mask_cg, [':START_ID', ':END_ID']]
        if not cg.empty:
            cg['row'] = cg[':START_ID'].map(chemical_mapping)
            cg['col'] = cg[':END_ID'].map(gene_mapping)
            cg = cg.dropna(subset=['row', 'col'])
            rows = cg['row'].astype(int).values
            cols = cg['col'].astype(int).values
            drug_gene_matrix[rows, cols] = 1

        # Process gene-chemical edges
        mask_gc = (chunk['node1_type'] == 'gene') & (chunk['node2_type'] == 'chemical')
        gc = chunk.loc[mask_gc, [':START_ID', ':END_ID']]
        if not gc.empty:
            gc['row'] = gc[':END_ID'].map(chemical_mapping)
            gc['col'] = gc[':START_ID'].map(gene_mapping)
            gc = gc.dropna(subset=['row', 'col'])
            rows = gc['row'].astype(int).values
            cols = gc['col'].astype(int).values
            drug_gene_matrix[rows, cols] = 1

        # Process disease-gene edges
        mask_dg = (chunk['node1_type'] == 'disease') & (chunk['node2_type'] == 'gene')
        dg = chunk.loc[mask_dg, [':START_ID', ':END_ID']]
        if not dg.empty:
            dg['row'] = dg[':START_ID'].map(disease_mapping)
            dg['col'] = dg[':END_ID'].map(gene_mapping)
            dg = dg.dropna(subset=['row', 'col'])
            rows = dg['row'].astype(int).values
            cols = dg['col'].astype(int).values
            disease_gene_matrix[rows, cols] = 1

        # Process gene-disease edges
        mask_gd = (chunk['node1_type'] == 'gene') & (chunk['node2_type'] == 'disease')
        gd = chunk.loc[mask_gd, [':START_ID', ':END_ID']]
        if not gd.empty:
            gd['row'] = gd[':END_ID'].map(disease_mapping)
            gd['col'] = gd[':START_ID'].map(gene_mapping)
            gd = gd.dropna(subset=['row', 'col'])
            rows = gd['row'].astype(int).values
            cols = gd['col'].astype(int).values
            disease_gene_matrix[rows, cols] = 1

    # Save association matrices (they will overwrite existing files)
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
