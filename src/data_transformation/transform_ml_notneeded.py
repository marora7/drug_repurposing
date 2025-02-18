"""
Transform the already exported nodes and edges into ML input format.

This script reads your gzipped nodes and edges CSV files,
and transforms them into separate files for the ML pipeline:
  - For LLM input: drug.csv, disease.csv, gene.csv
  - For GNN input:
      • Association matrices (saved as .npz): drug_disease.npz, drug_gene.npz, disease_gene.npz
      • Similarity matrices (saved as .npz): drug_drug.npz, disease_disease.npz, gene_gene.npz
"""

import os
import argparse
import logging
import pandas as pd
import numpy as np
import yaml

from src.utils.config_utils import load_config
from scipy.sparse import coo_matrix, save_npz

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
    Read edges CSV and create association matrices as sparse matrices:
      - drug_disease: chemicals (rows) x diseases (columns)
      - drug_gene: chemicals x genes
      - disease_gene: diseases x genes

    The matrices are saved as .npz files for efficient downstream loading.
    """
    logger.info("Loading edges from: %s", edges_file)
    
    n_chem = len(chemical_mapping)
    n_dis  = len(disease_mapping)
    n_gene = len(gene_mapping)

    # Lists to accumulate coordinate data for sparse matrices
    drug_disease_rows = []
    drug_disease_cols = []
    drug_gene_rows = []
    drug_gene_cols = []
    disease_gene_rows = []
    disease_gene_cols = []

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
            drug_disease_rows.extend(rows)
            drug_disease_cols.extend(cols)

        # Process disease-chemical edges
        mask_dc = (chunk['node1_type'] == 'disease') & (chunk['node2_type'] == 'chemical')
        dc = chunk.loc[mask_dc, [':START_ID', ':END_ID']]
        if not dc.empty:
            dc['row'] = dc[':END_ID'].map(chemical_mapping)
            dc['col'] = dc[':START_ID'].map(disease_mapping)
            dc = dc.dropna(subset=['row', 'col'])
            rows = dc['row'].astype(int).values
            cols = dc['col'].astype(int).values
            drug_disease_rows.extend(rows)
            drug_disease_cols.extend(cols)

        # Process chemical-gene edges
        mask_cg = (chunk['node1_type'] == 'chemical') & (chunk['node2_type'] == 'gene')
        cg = chunk.loc[mask_cg, [':START_ID', ':END_ID']]
        if not cg.empty:
            cg['row'] = cg[':START_ID'].map(chemical_mapping)
            cg['col'] = cg[':END_ID'].map(gene_mapping)
            cg = cg.dropna(subset=['row', 'col'])
            rows = cg['row'].astype(int).values
            cols = cg['col'].astype(int).values
            drug_gene_rows.extend(rows)
            drug_gene_cols.extend(cols)

        # Process gene-chemical edges
        mask_gc = (chunk['node1_type'] == 'gene') & (chunk['node2_type'] == 'chemical')
        gc = chunk.loc[mask_gc, [':START_ID', ':END_ID']]
        if not gc.empty:
            gc['row'] = gc[':END_ID'].map(chemical_mapping)
            gc['col'] = gc[':START_ID'].map(gene_mapping)
            gc = gc.dropna(subset=['row', 'col'])
            rows = gc['row'].astype(int).values
            cols = gc['col'].astype(int).values
            drug_gene_rows.extend(rows)
            drug_gene_cols.extend(cols)

        # Process disease-gene edges
        mask_dg = (chunk['node1_type'] == 'disease') & (chunk['node2_type'] == 'gene')
        dg = chunk.loc[mask_dg, [':START_ID', ':END_ID']]
        if not dg.empty:
            dg['row'] = dg[':START_ID'].map(disease_mapping)
            dg['col'] = dg[':END_ID'].map(gene_mapping)
            dg = dg.dropna(subset=['row', 'col'])
            rows = dg['row'].astype(int).values
            cols = dg['col'].astype(int).values
            disease_gene_rows.extend(rows)
            disease_gene_cols.extend(cols)

        # Process gene-disease edges
        mask_gd = (chunk['node1_type'] == 'gene') & (chunk['node2_type'] == 'disease')
        gd = chunk.loc[mask_gd, [':START_ID', ':END_ID']]
        if not gd.empty:
            gd['row'] = gd[':END_ID'].map(disease_mapping)
            gd['col'] = gd[':START_ID'].map(gene_mapping)
            gd = gd.dropna(subset=['row', 'col'])
            rows = gd['row'].astype(int).values
            cols = gd['col'].astype(int).values
            disease_gene_rows.extend(rows)
            disease_gene_cols.extend(cols)

    # Create sparse association matrices using COO format then convert to CSR.
    drug_disease_matrix = coo_matrix(
        (np.ones(len(drug_disease_rows), dtype=int),
         (np.array(drug_disease_rows), np.array(drug_disease_cols))),
        shape=(n_chem, n_dis)
    ).tocsr()

    drug_gene_matrix = coo_matrix(
        (np.ones(len(drug_gene_rows), dtype=int),
         (np.array(drug_gene_rows), np.array(drug_gene_cols))),
        shape=(n_chem, n_gene)
    ).tocsr()

    disease_gene_matrix = coo_matrix(
        (np.ones(len(disease_gene_rows), dtype=int),
         (np.array(disease_gene_rows), np.array(disease_gene_cols))),
        shape=(n_dis, n_gene)
    ).tocsr()

    # Save association matrices in sparse .npz format
    save_npz(os.path.join(output_dir, 'drug_disease.npz'), drug_disease_matrix)
    save_npz(os.path.join(output_dir, 'drug_gene.npz'), drug_gene_matrix)
    save_npz(os.path.join(output_dir, 'disease_gene.npz'), disease_gene_matrix)

    logger.info("Association matrices created (sparse): drug_disease %s, drug_gene %s, disease_gene %s",
                drug_disease_matrix.shape, drug_gene_matrix.shape, disease_gene_matrix.shape)
    return drug_disease_matrix, drug_gene_matrix, disease_gene_matrix


def compute_similarity_matrices(drug_disease_matrix, drug_gene_matrix, disease_gene_matrix, output_dir):
    """
    Compute similarity matrices as sparse matrices:
      - drug_drug: Dot product of drug_disease matrix with its transpose.
      - disease_disease: Dot product of the transpose of drug_disease matrix with itself.
      - gene_gene: Sum of similarity from drug_gene and disease_gene associations.

    The results are saved in sparse .npz format.
    """
    # Compute similarity matrices using sparse dot products.
    drug_drug_matrix = drug_disease_matrix.dot(drug_disease_matrix.T)
    save_npz(os.path.join(output_dir, 'drug_drug.npz'), drug_drug_matrix)

    disease_disease_matrix = drug_disease_matrix.T.dot(drug_disease_matrix)
    save_npz(os.path.join(output_dir, 'disease_disease.npz'), disease_disease_matrix)

    gene_similarity_from_drug = drug_gene_matrix.T.dot(drug_gene_matrix)
    gene_similarity_from_disease = disease_gene_matrix.T.dot(disease_gene_matrix)
    gene_gene_matrix = gene_similarity_from_drug + gene_similarity_from_disease
    save_npz(os.path.join(output_dir, 'gene_gene.npz'), gene_gene_matrix)

    logger.info("Similarity matrices computed (sparse).")
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
    
    # Transform edges and create sparse association matrices for GNN input
    drug_disease_matrix, drug_gene_matrix, disease_gene_matrix = transform_edges(
        edges_file, chemical_mapping, disease_mapping, gene_mapping, output_dir
    )
    
    # Compute similarity matrices as sparse matrices
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
