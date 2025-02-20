import os
os.environ["DGL_DISABLE_GRAPHBOLT"] = "1"
print("DGL_DISABLE_GRAPHBOLT =", os.environ.get("DGL_DISABLE_GRAPHBOLT"))

import sys
import types

# Create a dummy module for dgl.graphbolt
dummy_graphbolt = types.ModuleType("dgl.graphbolt")
dummy_graphbolt.load_graphbolt = lambda: None  # Override the load function to do nothing

# Insert the dummy module into sys.modules
sys.modules["dgl.graphbolt"] = dummy_graphbolt
sys.modules["dgl.graphbolt.__init__"] = dummy_graphbolt

import dgl
import torch
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from scipy.sparse import load_npz, csr_matrix

# Import configuration loader and seed setter
from src.utils.config_utils import load_config
from src.utils.seed_utils import set_seed


def sparse_topk_filtering(sparse_matrix: csr_matrix, k: int) -> np.array:
    """
    For each row in a CSR matrix, select the top-k nonzero entries
    (excluding the self-loop) and return their (row, col) indices.
    
    This avoids converting the entire matrix to dense.
    """
    rows = []
    cols = []
    for i in range(sparse_matrix.shape[0]):
        start_ptr = sparse_matrix.indptr[i]
        end_ptr = sparse_matrix.indptr[i + 1]
        # Get the indices and data for the current row
        row_indices = sparse_matrix.indices[start_ptr:end_ptr]
        row_data = sparse_matrix.data[start_ptr:end_ptr]
        
        # Exclude self-loop if present
        mask = row_indices != i
        if np.any(mask):
            row_indices = row_indices[mask]
            row_data = row_data[mask]
        else:
            # if no non-self entries, skip this row
            continue

        if len(row_data) == 0:
            continue
        
        if len(row_data) > k:
            # Get indices of the top k values in this row
            topk_idx = np.argpartition(row_data, -k)[-k:]
            selected_cols = row_indices[topk_idx]
        else:
            selected_cols = row_indices
        
        rows.extend([i] * len(selected_cols))
        cols.extend(selected_cols)
    return np.vstack((rows, cols)).T


def meta_path_instance(config, chem_id: int, dis_id: int, links: dict, k: int):
    """
    Generate a bag of meta-path instances for a given chemical-disease pair.

    Schemas:
      - A: Direct: [chem_id, chem_id, dis_id, dis_id]
      - B: Chemical -> Chemical -> Disease -> Disease
      - C: Chemical -> Gene -> Disease
      - D: Chemical -> Chemical -> Gene -> Disease
      - E: Chemical -> Gene -> Disease -> Disease

    The bag is padded or trimmed to a fixed length: target_len = k*(k+2) + 1.
    """
    mpi = []
    # Schema A: Direct meta path
    mpi.append([chem_id, chem_id, dis_id, dis_id])
    
    # Schema B: Chemical -> Chemical -> Disease -> Disease
    if 'chemical-chemical' in links and 'disease-disease' in links:
        chem_neighbors = links['chemical-chemical'][links['chemical-chemical'][:, 0] == chem_id][:, 1]
        dis_neighbors = links['disease-disease'][links['disease-disease'][:, 0] == dis_id][:, 1]
        for c in chem_neighbors:
            mpi.append([chem_id, int(c), dis_id, dis_id])
        for d in dis_neighbors:
            mpi.append([chem_id, chem_id, int(d), dis_id])
        for c in chem_neighbors:
            for d in dis_neighbors:
                mpi.append([chem_id, int(c), int(d), dis_id])
    
    # Schema C: Chemical -> Gene -> Disease
    if 'chemical-gene' in links and 'gene-disease' in links:
        chem_gene = links['chemical-gene'][links['chemical-gene'][:, 0] == chem_id][:, 1]
        for g in chem_gene:
            gene_dis = links['gene-disease'][links['gene-disease'][:, 0] == int(g)][:, 1]
            if dis_id in gene_dis:
                mpi.append([chem_id, int(g), dis_id, dis_id])
    
    # Schema D: Chemical -> Chemical -> Gene -> Disease
    if 'chemical-chemical' in links and 'chemical-gene' in links and 'gene-disease' in links:
        chem_neighbors = links['chemical-chemical'][links['chemical-chemical'][:, 0] == chem_id][:, 1]
        for c in chem_neighbors:
            gene_neighbors = links['chemical-gene'][links['chemical-gene'][:, 0] == int(c)][:, 1]
            for g in gene_neighbors:
                gene_dis = links['gene-disease'][links['gene-disease'][:, 0] == int(g)][:, 1]
                if dis_id in gene_dis:
                    mpi.append([chem_id, int(c), int(g), dis_id])
    
    # Schema E: Chemical -> Gene -> Disease -> Disease
    if 'chemical-gene' in links and 'disease-disease' in links and 'gene-disease' in links:
        gene_neighbors = links['chemical-gene'][links['chemical-gene'][:, 0] == chem_id][:, 1]
        dis_neighbors = links['disease-disease'][links['disease-disease'][:, 0] == dis_id][:, 1]
        for g in gene_neighbors:
            gene_dis = links['gene-disease'][links['gene-disease'][:, 0] == int(g)][:, 1]
            for d in dis_neighbors:
                if int(d) in gene_dis:
                    mpi.append([chem_id, int(g), int(d), dis_id])
    
    target_len = k * (k + 2) + 1
    if len(mpi) < target_len:
        for _ in range(target_len - len(mpi)):
            mpi.append(random.choice(mpi))
    elif len(mpi) > target_len:
        mpi = mpi[:target_len]
    return mpi


def load_data(config):
    """
    Load dataset and construct a heterogeneous graph with three node types:
      - chemical, disease, and gene.
    
    Expects the following NPZ files in config['ml_input_dir']:
      - drug_drug.npz      : Intra-type similarity matrix for chemicals.
      - disease_disease.npz: Intra-type similarity matrix for diseases.
      - gene_gene.npz      : Intra-type similarity matrix for genes.
      - drug_disease.npz   : Binary association matrix (chemical-disease).
      - drug_gene.npz      : Binary association matrix (chemical-gene).
      - disease_gene.npz   : Binary association matrix (disease-gene).
    
    Also generates meta-path instances for each chemical-disease pair.
    
    Returns:
      g     : DGL heterograph.
      data  : NumPy array of meta-path instance bags.
      label : NumPy array of binary labels from the chemical-disease matrix.
    """
    dataset_dir = config.get("ml_input_dir", "data/processed/ml_input")
    k = config.get("k", 15)

    # --- Load intra-type similarity matrices (sparse) and apply top-k filtering ---
    chem_chem_sparse = load_npz(os.path.join(dataset_dir, "drug_drug.npz"))
    chem_chem_link = sparse_topk_filtering(chem_chem_sparse, k)

    dis_dis_sparse = load_npz(os.path.join(dataset_dir, "disease_disease.npz"))
    dis_dis_link = sparse_topk_filtering(dis_dis_sparse, k)

    gene_gene_sparse = load_npz(os.path.join(dataset_dir, "gene_gene.npz"))
    gene_gene_link = sparse_topk_filtering(gene_gene_sparse, k)

    # --- Load cross-type binary association matrices using nonzero ---
    chem_dis_sparse = load_npz(os.path.join(dataset_dir, "drug_disease.npz"))
    chem_dis_link = np.array(chem_dis_sparse.nonzero()).T
    dis_chem_link = np.array(chem_dis_sparse.T.nonzero()).T

    chem_gene_sparse = load_npz(os.path.join(dataset_dir, "drug_gene.npz"))
    chem_gene_link = np.array(chem_gene_sparse.nonzero()).T
    gene_chem_link = np.array(chem_gene_sparse.T.nonzero()).T

    dis_gene_sparse = load_npz(os.path.join(dataset_dir, "disease_gene.npz"))
    dis_gene_link = np.array(dis_gene_sparse.nonzero()).T
    gene_dis_link = np.array(dis_gene_sparse.T.nonzero()).T

    # Build the links dictionary for meta-path generation
    links = {
        'chemical-chemical': chem_chem_link,
        'disease-disease': dis_dis_link,
        'gene-gene': gene_gene_link,
        'chemical-disease': chem_dis_link,
        'disease-chemical': dis_chem_link,
        'chemical-gene': chem_gene_link,
        'gene-chemical': gene_chem_link,
        'disease-gene': dis_gene_link,
        'gene-disease': gene_dis_link,
    }

    # --- Build the heterograph dictionary ---
    # Note: We use the top-k filtered intra-type links and the binary cross-type links.
    graph_data = {
        ('chemical', 'chemical', 'chemical'): (
            torch.tensor(chem_chem_link[:, 0], dtype=torch.int64),
            torch.tensor(chem_chem_link[:, 1], dtype=torch.int64)
        ),
        ('disease', 'disease', 'disease'): (
            torch.tensor(dis_dis_link[:, 0], dtype=torch.int64),
            torch.tensor(dis_dis_link[:, 1], dtype=torch.int64)
        ),
        ('gene', 'gene', 'gene'): (
            torch.tensor(gene_gene_link[:, 0], dtype=torch.int64),
            torch.tensor(gene_gene_link[:, 1], dtype=torch.int64)
        ),
        ('chemical', 'chemical-disease', 'disease'): (
            torch.tensor(chem_dis_link[:, 0], dtype=torch.int64),
            torch.tensor(chem_dis_link[:, 1], dtype=torch.int64)
        ),
        ('disease', 'disease-chemical', 'chemical'): (
            torch.tensor(dis_chem_link[:, 0], dtype=torch.int64),
            torch.tensor(dis_chem_link[:, 1], dtype=torch.int64)
        ),
        ('chemical', 'chemical-gene', 'gene'): (
            torch.tensor(chem_gene_link[:, 0], dtype=torch.int64),
            torch.tensor(chem_gene_link[:, 1], dtype=torch.int64)
        ),
        ('gene', 'gene-chemical', 'chemical'): (
            torch.tensor(gene_chem_link[:, 0], dtype=torch.int64),
            torch.tensor(gene_chem_link[:, 1], dtype=torch.int64)
        ),
        ('disease', 'disease-gene', 'gene'): (
            torch.tensor(dis_gene_link[:, 0], dtype=torch.int64),
            torch.tensor(dis_gene_link[:, 1], dtype=torch.int64)
        ),
        ('gene', 'gene-disease', 'disease'): (
            torch.tensor(gene_dis_link[:, 0], dtype=torch.int64),
            torch.tensor(gene_dis_link[:, 1], dtype=torch.int64)
        ),
    }
    g = dgl.heterograph(graph_data)

    # --- Construct node features ---
    # Instead of using huge similarity matrices, we assign each node a low-dimensional random feature vector.
    feature_dim = config.get("node_feature_dim", 16)
    num_chem = chem_chem_sparse.shape[0]
    num_dis = dis_dis_sparse.shape[0]
    num_gene = gene_gene_sparse.shape[0]

    chem_feature = torch.randn(num_chem, feature_dim)
    dis_feature = torch.randn(num_dis, feature_dim)
    gene_feature = torch.randn(num_gene, feature_dim)
    g.nodes['chemical'].data['h'] = chem_feature
    g.nodes['disease'].data['h'] = dis_feature
    g.nodes['gene'].data['h'] = gene_feature

    # --- Generate meta-path instances for each chemical-disease pair ---
    # Instead of converting the whole chemical-disease association matrix to dense,
    # we build a lookup set for label values.
    chem_dis_edges_set = set(map(tuple, chem_dis_link))
    data = []
    label = []
    num_chem, num_dis = chem_dis_sparse.shape

    temp_dir = f"{dataset_dir}_temp_{k}k"
    if os.path.exists(temp_dir):
        data = np.load(os.path.join(temp_dir, "data.npy"), allow_pickle=True)
        label = np.load(os.path.join(temp_dir, "label.npy"), allow_pickle=True)
    else:
        os.mkdir(temp_dir)
        desc = f"Chemical x Disease ({num_chem} x {num_dis})"
        for chem_id in tqdm(range(num_chem), desc=desc):
            for dis_id in range(num_dis):
                data.append(meta_path_instance(config, chem_id, dis_id, links, k))
                # Use the lookup set to determine if an edge exists.
                label_val = 1 if (chem_id, dis_id) in chem_dis_edges_set else 0
                label.append(label_val)
        data = np.array(data, dtype=object)
        label = np.array(label)
        np.save(os.path.join(temp_dir, "data.npy"), data, allow_pickle=True)
        np.save(os.path.join(temp_dir, "label.npy"), label, allow_pickle=True)
    
    return g, data, label


def remove_graph(g, test_chem_ids, test_dis_ids):
    """
    Remove edges between specified chemical and disease IDs from the heterograph.
    """
    etype = ('chemical', 'chemical-disease', 'disease')
    edges_id = g.edge_ids(torch.tensor(test_chem_ids), torch.tensor(test_dis_ids), etype=etype)
    g = dgl.remove_edges(g, edges_id, etype=etype)
    etype = ('disease', 'disease-chemical', 'chemical')
    edges_id = g.edge_ids(torch.tensor(test_dis_ids), torch.tensor(test_chem_ids), etype=etype)
    g = dgl.remove_edges(g, edges_id, etype=etype)
    return g


def get_data_loaders(data, batch_size, shuffle, drop=False):
    """Return a DataLoader for training/testing."""
    return DataLoader(data, batch_size=batch_size, shuffle=shuffle, drop_last=drop)


if __name__ == "__main__":
    # Load configuration (adjust the path as needed)
    config = load_config("src/config/config.yaml")
    # Optionally, set seed for reproducibility
    set_seed(config.get("seed", 42))

    # Load data and build the heterograph, meta-path instances, and labels
    g, meta_path_data, labels = load_data(config)

    # Print out some details about the outputs
    print("=== Execution Complete ===")
    print("Heterograph Summary:")
    print(g)
    print("Number of meta-path instance bags:", len(meta_path_data))
    print("Shape of labels array:", labels.shape)
