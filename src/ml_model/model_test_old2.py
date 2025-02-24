"""
Link Prediction Script:
This script loads pre-trained TransE embeddings along with entity and relation data,
and performs link prediction for one or more head and tail entity pairs.
For each relation, it computes the predicted tail embedding as:
    head_embedding + relation_embedding,
and then calculates the L2 distance to the given tail embedding.
A positive score is computed as:
    score = 1 / (distance + epsilon)
so that a smaller distance yields a higher score.
The relations are sorted in descending order by score, and the top_k relations are returned.
Configuration parameters (file paths, top_k, etc.) are read from config.yaml.
"""

import os
import numpy as np
import argparse
import pandas as pd
import csv
import torch
from src.utils.config_utils import load_config

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

def predict_relations(head_idx, tail_idx, entity_emb, relation_emb, top_k=10):
    """
    Given a head and tail entity index, computes a score for each relation using:
        score = 1 / (|| (head_embedding + relation_embedding) - tail_embedding || + epsilon)
    (i.e. a positive score where higher is better).
    The function sorts the scores in descending order and returns the indices of the top_k relations,
    along with the full score array.
    """
    head_vec = entity_emb[head_idx]   # shape: (d,)
    tail_vec = entity_emb[tail_idx]   # shape: (d,)
    # Compute predicted tail embeddings for all relations (broadcast addition)
    predicted = head_vec + relation_emb  # shape: (num_relations, d)
    differences = predicted - tail_vec   # shape: (num_relations, d)
    distances = torch.norm(differences, dim=1)  # L2 distances for all relations
    epsilon = 1e-8
    scores = 1.0 / (distances + epsilon)  # higher score for smaller distance
    # Sort by descending score and filter top_k after sorting
    sorted_indices = torch.argsort(scores, descending=True)
    top_indices = sorted_indices[:top_k]
    # Convert to numpy arrays for further processing/printing
    scores = scores.cpu().numpy()
    top_indices = top_indices.cpu().numpy()
    return top_indices, scores

def print_predictions(head_entity, tail_entity, relation_list, top_relation_indices, scores):
    """
    Prints the prediction results in a table format using Pandas.
    """
    results = []
    for rank, rel_idx in enumerate(top_relation_indices, start=1):
        relation = relation_list[rel_idx]
        score = scores[rel_idx]  # positive score where higher is better
        results.append({
            "Rank": rank,
            "Head Entity": head_entity,
            "Tail Entity": tail_entity,
            "Predicted Relation": relation,
            "Score": f"{score:.4f}"
        })
    
    df = pd.DataFrame(results)
    print("\nPredictions:")
    print(df.to_string(index=False))

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
        description="Link prediction for one or more head and tail entity pairs using config.yaml"
    )
    parser.add_argument("head_entities", type=str,
                        help="Comma-separated list of head entity strings (e.g., 'Chemical::DRUG123,Chemical::DRUG456')")
    parser.add_argument("tail_entities", type=str,
                        help="Comma-separated list of tail entity strings (e.g., 'Disease::DISEASE456,Disease::DISEASE789')")
    args = parser.parse_args()
    
    # Parse the provided entity strings into lists
    head_list = parse_entities(args.head_entities)
    tail_list = parse_entities(args.tail_entities)
    
    # If one list is length 1 and the other is longer, replicate the single entity accordingly
    if len(head_list) == 1 and len(tail_list) > 1:
        head_list = head_list * len(tail_list)
    elif len(tail_list) == 1 and len(head_list) > 1:
        tail_list = tail_list * len(head_list)
    elif len(head_list) != len(tail_list):
        print("Error: When providing multiple entities, the number of head and tail entities must match (or one of them must be a single value).")
        return
    
    # Determine the path to config.yaml relative to this file (assumes config.yaml is in the project root)
    CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
    config = load_config(CONFIG_PATH)
    
    # Extract parameters from the configuration
    nodes_file = config["nodes_file"]
    train_file = config["train_file"]
    entity_embedding_path = config["entity_embedding_path"]
    relation_embedding_path = config["relation_embedding_path"]
    top_k = config.get("top_k", 10)
    
    # Set up the device: use CUDA if available, else CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load embeddings and move them to the selected device
    print("Loading embeddings...")
    entity_emb, relation_emb = load_embeddings(entity_embedding_path, relation_embedding_path, device)
    
    # Load ordered entity list from nodes file
    print("Loading entities from nodes file...")
    entity_list = load_ordered_entities(nodes_file)
    
    # Load ordered relation list from training file
    print("Extracting relations from training file...")
    relation_list = load_ordered_relations(train_file)
    
    # Process each head-tail pair
    for head_entity, tail_entity in zip(head_list, tail_list):
        try:
            head_idx = entity_list.index(head_entity)
        except ValueError:
            print(f"Head entity '{head_entity}' not found in nodes file. Skipping this pair.")
            continue
        try:
            tail_idx = entity_list.index(tail_entity)
        except ValueError:
            print(f"Tail entity '{tail_entity}' not found in nodes file. Skipping this pair.")
            continue

        print(f"\nRunning link prediction for head entity '{head_entity}' and tail entity '{tail_entity}'...\n")
        # Predict and rank relations based on the positive score (descending order)
        top_relation_indices, scores = predict_relations(head_idx, tail_idx, entity_emb, relation_emb, top_k=top_k)
        # Display the results with header "Score"
        print_predictions(head_entity, tail_entity, relation_list, top_relation_indices, scores)
        print("-" * 40)

if __name__ == "__main__":
    main()
