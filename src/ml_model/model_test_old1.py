"""
Link Prediction Script:
This script loads pre-trained TransE embeddings along with entity and relation data, 
and performs link prediction for a given head entity across all relations. 
Configuration parameters (file paths, top_k, etc.) are read from config.yaml.
"""

import os
import numpy as np
import argparse
import pandas as pd
import csv
import sys
import pathlib
from src.utils.config_utils import load_config

def load_embeddings(entity_path, relation_path):
    """
    Loads the entity and relation embedding matrices from .npy files.
    """
    if not os.path.exists(entity_path):
        raise FileNotFoundError(f"Entity embedding file not found: {entity_path}")
    if not os.path.exists(relation_path):
        raise FileNotFoundError(f"Relation embedding file not found: {relation_path}")
    
    entity_embeddings = np.load(entity_path)
    relation_embeddings = np.load(relation_path)
    return entity_embeddings, relation_embeddings

def load_ordered_entities(nodes_file):
    """
    Loads entities from the nodes CSV (or CSV.GZ) file and creates an ordered list.
    Assumes the CSV has columns 'Id:ID' and ':LABEL' and that the training entity string was
    generated as "<:LABEL>::<Id:ID>".
    """
    compression = 'gzip' if nodes_file.endswith('.gz') else None
    df = pd.read_csv(nodes_file, compression=compression)
    entity_list = [f"{row[':LABEL']}::{row['Id:ID']}" for _, row in df.iterrows()]
    return entity_list

def load_ordered_relations(train_file):
    """
    Reads the training TSV file (assumed columns: head, relation, tail, score) and returns
    an ordered list of unique relation strings (in the order they first appear).
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

def predict_tail_entities(head_idx, relation_idx, entity_emb, relation_emb, top_k=2):
    """
    For a given head index and relation index, computes the predicted tail embedding
    as: head_embedding + relation_embedding.
    Then, it computes the L2 distance between the predicted vector and all entity embeddings.
    Returns the indices of the top_k closest entities and their distances.
    """
    head_vec = entity_emb[head_idx]
    rel_vec = relation_emb[relation_idx]
    pred_tail = head_vec + rel_vec
    distances = np.linalg.norm(entity_emb - pred_tail, axis=1)
    top_indices = np.argsort(distances)[:top_k]
    return top_indices, distances

def main():
    parser = argparse.ArgumentParser(
        description="Link prediction for a given head entity across all relations using config.yaml"
    )
    parser.add_argument("entity_str", type=str,
                        help="Input head entity string (e.g., 'Disease::MESH:D007752')")
    args = parser.parse_args()
    
    # Determine the path to config.yaml relative to this file (assumes config.yaml is in the project root)
    CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
    config = load_config(CONFIG_PATH)
    
    # Extract parameters from the configuration
    nodes_file = config["nodes_file"]
    train_file = config["train_file"]
    entity_embedding_path = config["entity_embedding_path"]
    relation_embedding_path = config["relation_embedding_path"]
    top_k = config.get("top_k", 2)
    
    # Load embeddings
    print("Loading embeddings...")
    entity_emb, relation_emb = load_embeddings(entity_embedding_path, relation_embedding_path)
    
    # Load ordered entity list from nodes file
    print("Loading entities from nodes file...")
    entity_list = load_ordered_entities(nodes_file)
    
    # Load ordered relation list from training file
    print("Extracting relations from training file...")
    relation_list = load_ordered_relations(train_file)
    
    # Find index for the given head entity string
    try:
        head_idx = entity_list.index(args.entity_str)
    except ValueError:
        print(f"Entity '{args.entity_str}' not found in nodes file.")
        return
    
    print(f"\nRunning link prediction for head entity '{args.entity_str}' across all relations...\n")
    # For each relation, perform prediction and print results
    for rel_idx, relation in enumerate(relation_list):
        try:
            top_indices, distances = predict_tail_entities(head_idx, rel_idx, entity_emb, relation_emb, top_k=top_k)
            print(f"Relation: {relation}")
            for idx in top_indices:
                tail_entity = entity_list[idx]
                score = distances[idx]
                print(f"  {tail_entity}  (score: {score:.4f})")
            print("-" * 40)
        except Exception as e:
            print(f"Error processing relation '{relation}': {e}")

if __name__ == "__main__":
    main()
