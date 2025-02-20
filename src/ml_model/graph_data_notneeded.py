import os
import dgl
import torch
import random
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

# Import configuration loader
from src.utils.config_utils import load_config
from src.utils.seed_utils import set_seed

def topk_filtering(matrix: np.array, k: int) -> np.array:
    """
    For each row, select the top-k similarities (excluding the self-loop)
    and set them to 1. Return the indices (row, col) where the matrix equals 1.
    """
    for i in range(len(matrix)):
        sorted_idx = np.argpartition(matrix[i], -k - 1)
        matrix[i, sorted_idx[-k - 1:-1]] = 1
    return np.array(np.where(matrix == 1)).T

def meta_path_instance(config, chem_id: int, dis_id: int, links: dict, k: int):
    """
    Generate a bag of meta-path instances.
    
    This function generates meta-paths using several schemas:
      - A: Direct: [chem_id, chem_id, dis_id, dis_id]
      - B: Chemical -> Chemical -> Disease -> Disease (using intra-type similarities)
      - C: Chemical -> Gene -> Disease (via gene association)
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
    
    Expects the following CSV files in config['dataset_dir']:
      - drug_drug.csv      : Intra-type similarity matrix for chemicals.
      - disease_disease.csv: Intra-type similarity matrix for diseases.
      - gene_gene.csv      : Intra-type similarity matrix for genes.
      - drug_disease.csv   : Binary association matrix (chemical-disease).
      - drug_gene.csv      : Binary association matrix (chemical-gene).
      - disease_gene.csv   : Binary association matrix (disease-gene).
    
    Also generates meta-path instances for each chemical-disease pair.
    
    Returns:
      g     : DGL heterograph.
      data  : numpy array of meta-path instance bags.
      label : numpy array of binary labels from the chemical-disease matrix.
    """
    dataset_dir = config.get("ml_input_dir", "data/processed/ml_input")
    k = config.get("k", 15)

    # Load intra-type similarity matrices and apply top-k filtering
    chem_chem = pd.read_csv(os.path.join(dataset_dir, "drug_drug.csv"), header=None)
    chem_chem_link = topk_filtering(chem_chem.values, k)

    dis_dis = pd.read_csv(os.path.join(dataset_dir, "disease_disease.csv"), header=None)
    dis_dis_link = topk_filtering(dis_dis.values, k)

    gene_gene = pd.read_csv(os.path.join(dataset_dir, "gene_gene.csv"), header=None)
    gene_gene_link = topk_filtering(gene_gene.values, k)

    # Load cross-type binary association matrices
    chem_dis = pd.read_csv(os.path.join(dataset_dir, "drug_disease.csv"), header=None)
    chem_dis_link = np.array(np.where(chem_dis == 1)).T
    dis_chem_link = np.array(np.where(chem_dis.T == 1)).T

    chem_gene = pd.read_csv(os.path.join(dataset_dir, "drug_gene.csv"), header=None)
    chem_gene_link = np.array(np.where(chem_gene == 1)).T
    gene_chem_link = np.array(np.where(chem_gene.T == 1)).T

    dis_gene = pd.read_csv(os.path.join(dataset_dir, "disease_gene.csv"), header=None)
    dis_gene_link = np.array(np.where(dis_gene == 1)).T
    gene_dis_link = np.array(np.where(dis_gene.T == 1)).T

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

    # Build the heterograph dictionary
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

    # Construct node features (a simple approach for illustration)
    chem_feature = np.hstack((chem_chem.values, np.zeros(chem_dis.shape)))
    dis_feature = np.hstack((np.zeros(chem_dis.T.shape), dis_dis.values))
    gene_feature = np.hstack((np.zeros(chem_gene.T.shape), gene_gene.values))
    g.nodes['chemical'].data['h'] = torch.from_numpy(chem_feature).to(torch.float32)
    g.nodes['disease'].data['h'] = torch.from_numpy(dis_feature).to(torch.float32)
    g.nodes['gene'].data['h'] = torch.from_numpy(gene_feature).to(torch.float32)

    # Generate meta-path instances for each chemical-disease pair
    data = []
    label = []
    num_chem, num_dis = chem_dis.shape
    temp_dir = f"{dataset_dir}_temp_{k}k"
    if os.path.exists(temp_dir):
        data = np.load(os.path.join(temp_dir, "data.npy"))
        label = np.load(os.path.join(temp_dir, "label.npy"))
    else:
        os.mkdir(temp_dir)
        desc = f"Chemical x Disease ({num_chem} x {num_dis})"
        for chem_id in tqdm(range(num_chem), desc=desc):
            for dis_id in range(num_dis):
                data.append(meta_path_instance(config, chem_id, dis_id, links, k))
                label.append(int(chem_dis.iloc[chem_id, dis_id]))
        data = np.array(data)
        label = np.array(label)
        np.save(os.path.join(temp_dir, "data.npy"), data)
        np.save(os.path.join(temp_dir, "label.npy"), label)
    
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
