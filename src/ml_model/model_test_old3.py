import os
import numpy as np
import argparse
import pandas as pd
import csv
import torch
from src.utils.config_utils import load_config
import itertools

def load_embeddings(entity_path, relation_path, device):
    """
    Loads the entity and relation embedding matrices from .npy files,
    converts them to PyTorch tensors, and moves them to the specified device.
    """
    if not os.path.exists(entity_path):
        raise FileNotFoundError(f"Entity embedding file not found: {entity_path}")
    if not os.path.exists(relation_path):
        raise FileNotFoundError(f"Relation embedding file not found: {relation_path}")
    
    entity_embeddings = torch.from_numpy(np.load(entity_path)).float().to(device)
    relation_embeddings = torch.from_numpy(np.load(relation_path)).float().to(device)
    return entity_embeddings, relation_embeddings

def load_ordered_entities(nodes_file):
    """
    Loads entities from the nodes CSV (or CSV.GZ) file and creates an ordered list.
    Assumes the CSV has columns 'Id:ID' and ':LABEL' and that the training entity
    string was generated as '<:LABEL>::<Id:ID>'.
    """
    compression = 'gzip' if nodes_file.endswith('.gz') else None
    df = pd.read_csv(nodes_file, compression=compression)
    entity_list = [f"{row[':LABEL']}::{row['Id:ID']}" for _, row in df.iterrows()]
    return entity_list

def load_ordered_relations(train_file):
    """
    Reads the training TSV file (assumed columns: head, relation, tail, score)
    and returns an ordered list of unique relation strings (in the order they first appear).
    """
    relations = []
    seen = set()
    with open(train_file, "r") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) < 3:
                continue
            relation = row[1]
            if relation not in seen:
                seen.add(relation)
                relations.append(relation)
    return relations

def compute_distance(head_idx, relation_idx, tail_idx, entity_emb, relation_emb):
    """
    Given the indices for a head entity, relation, and tail entity,
    computes the L2 distance = ||head_embedding + relation_embedding - tail_embedding||.
    A smaller distance means the triple is closer in the embedding space.
    """
    head_vec = entity_emb[head_idx]
    rel_vec  = relation_emb[relation_idx]
    tail_vec = entity_emb[tail_idx]
    
    distance = torch.norm(head_vec + rel_vec - tail_vec, p=2)
    return distance.item()

def parse_entities(entity_string):
    """
    Parses a comma-separated string of entities into a list.
    If no comma is found, returns a list with a single entity.
    """
    if "," in entity_string:
        return [e.strip() for e in entity_string.split(",")]
    else:
        return [entity_string.strip()]

def main():
    parser = argparse.ArgumentParser(
        description="Link prediction for (chemical, relation, disease) triplets using config.yaml"
    )
    # Changed to plural: --diseases, --relations, --chemicals
    parser.add_argument("--diseases", type=str, required=True,
                        help="Comma-separated list of diseases, e.g., 'Disease::Disease:MESH:D003920' or 'Disease::Disease:MESH:D003920,Disease::Disease:MESH:DXXXX'")
    parser.add_argument("--relations", type=str, required=True,
                        help="Comma-separated list of relations, e.g., 'treat:Chemical:Disease' or 'treat:Chemical:Disease,causes:Chemical:Disease'")
    parser.add_argument("--chemicals", type=str, required=True,
                        help="Comma-separated list of chemicals, e.g. 'Chemical::Chemical:MESH:D007328,Chemical::Chemical:MESH:D008687'")
    
    args = parser.parse_args()
    
    # Determine the path to config.yaml relative to this file
    CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
    config = load_config(CONFIG_PATH)
    
    # Extract parameters from the configuration
    nodes_file = config["nodes_file"]
    train_file = config["train_file"]
    entity_embedding_path = config["entity_embedding_path"]
    relation_embedding_path = config["relation_embedding_path"]
    
    # Set up the device: use CUDA if available, else CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load embeddings
    print("Loading embeddings...")
    entity_emb, relation_emb = load_embeddings(entity_embedding_path, relation_embedding_path, device)
    
    # Load entity and relation lists
    print("Loading entities from nodes file...")
    entity_list = load_ordered_entities(nodes_file)
    
    print("Extracting relations from training file...")
    relation_list = load_ordered_relations(train_file)
    
    # Parse the arguments as lists
    diseases_list  = parse_entities(args.diseases)
    relations_list = parse_entities(args.relations)
    chemicals_list = parse_entities(args.chemicals)
    
    results = []
    
    # Generate all combinations of (disease, relation, chemical)
    for disease_str, relation_str, chem_str in itertools.product(diseases_list, relations_list, chemicals_list):
        # Find indices for each
        try:
            disease_idx = entity_list.index(disease_str)
        except ValueError:
            print(f"WARNING: disease '{disease_str}' not found in nodes file. Skipping.")
            continue
        
        try:
            relation_idx = relation_list.index(relation_str)
        except ValueError:
            print(f"WARNING: relation '{relation_str}' not found in training relations. Skipping.")
            continue
        
        try:
            chem_idx = entity_list.index(chem_str)
        except ValueError:
            print(f"WARNING: chemical '{chem_str}' not found in nodes file. Skipping.")
            continue
        
        # Compute the triple distance
        triple_distance = compute_distance(chem_idx, relation_idx, disease_idx, entity_emb, relation_emb)
        results.append({
            "Disease": disease_str,
            "Relation": relation_str,
            "Chemical": chem_str,
            "Distance": triple_distance
        })
    
    # Sort results in ascending order by distance (lower is better)
    results.sort(key=lambda x: x["Distance"])
    
    # Print the results in a table
    df = pd.DataFrame(results)
    print("\nTriple Distances for (Chemical, Relation, Disease):\n")
    print(df.to_string(index=False, float_format="%.4f"))

if __name__ == "__main__":
    main()
