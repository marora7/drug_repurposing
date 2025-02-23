import os
import csv
import numpy as np
import pandas as pd
from time import time
from tqdm import tqdm
from src.utils.config_utils import load_config

# Determine the path to config.yaml relative to this file (assumes config.yaml is in the project root)
CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
config = load_config(CONFIG_PATH)

# Input file paths (relative to the project root)
nodes_file = os.path.join("data", "exports", "nodes.csv.gz")
edges_file = os.path.join("data", "exports", "edges.csv.gz")

# Output directory from config.yaml (e.g., data/processed/ml_input)
output_dir = config["data_dir"]
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# The generated data file will be stored in the output directory
generated_data_filepath = os.path.join(output_dir, "train.tsv")

def parse_node_id(node_id):
    """
    Parses a node id string into its label and identifier (CUI).
    For example:
      - "Disease:MESH:C537159" -> ("Disease", "MESH:C537159")
      - "Gene:643669" -> ("Gene", "643669")
    """
    parts = node_id.split(":")
    label = parts[0]
    if len(parts) > 2:
        cui = ":".join(parts[1:])
    elif len(parts) == 2:
        cui = parts[1]
    else:
        cui = node_id  # fallback if format is unexpected
    return label, cui

def generate_data():
    """
    Reads nodes and edges from CSV files, creates numeric mappings for nodes and relations,
    and writes the output as numeric IDs to train.tsv.
    """
    print("Loading nodes...")
    nodes_df = pd.read_csv(nodes_file, compression="gzip")
    
    # Build a lookup from original node id to its label and name
    node_map = {}
    for _, row in nodes_df.iterrows():
        node_map[row["Id:ID"]] = {"label": row[":LABEL"], "name": row["name"]}
    
    print("Loading edges...")
    edges_df = pd.read_csv(edges_file, compression="gzip")
    total_edges = len(edges_df)
    print("Total edges in graph:", total_edges)

    # First pass: build sets of unique nodes and relations based on their string representations
    node_set = set()
    rel_set = set()
    edge_rows = []  # temporary storage for processed edges

    for _, edge in tqdm(edges_df.iterrows(), total=total_edges, desc="Processing edges"):
        start_id = edge[":START_ID"]
        end_id = edge[":END_ID"]
        relation = edge[":TYPE"]
        score = int(edge["pmcount:int"])  # ensure score is an integer

        # Process start node details
        if start_id in node_map:
            entity1_label = node_map[start_id]["label"]
        else:
            # Fall back to parsing if node_map is missing this node
            entity1_label, _ = parse_node_id(start_id)
        _, entity1_cui = parse_node_id(start_id)
        entity1_str = f"{entity1_label}::{entity1_cui}"

        # Process end node details
        if end_id in node_map:
            entity2_label = node_map[end_id]["label"]
        else:
            entity2_label, _ = parse_node_id(end_id)
        _, entity2_cui = parse_node_id(end_id)
        entity2_str = f"{entity2_label}::{entity2_cui}"

        # Build a relation string similar to your original code.
        relation_str = f"ALPHAMELD::{relation}:{entity1_label}:{entity2_label}"

        # Record the string representations
        node_set.add(entity1_str)
        node_set.add(entity2_str)
        rel_set.add(relation_str)

        edge_rows.append((entity1_str, relation_str, entity2_str, score))

    # Create mappings from string to unique integer id (ensuring int64)
    node2id = {node: idx for idx, node in enumerate(sorted(node_set))}
    rel2id = {rel: idx for idx, rel in enumerate(sorted(rel_set))}

    print(f"Total unique nodes: {len(node2id)}")
    print(f"Total unique relations: {len(rel2id)}")

    start_time = time()
    with open(generated_data_filepath, "w", newline="") as tsvfile:
        csv_writer = csv.writer(tsvfile, delimiter="\t")
        # Write each triple as numeric IDs
        for entity1_str, relation_str, entity2_str, score in edge_rows:
            head = node2id[entity1_str]
            tail = node2id[entity2_str]
            rel = rel2id[relation_str]
            row = [head, rel, tail, score]
            csv_writer.writerow(row)
    
    print("Time taken to generate data: {:.2f} seconds".format(time() - start_time))

def train_test_val_split():
    """
    Splits the generated train.tsv data into train, test, and validation sets
    (90%, 5%, 5% respectively) and writes them into the output directory.
    """
    print("Splitting generated data into train, test, and val files...")
    df = pd.read_csv(generated_data_filepath, sep="\t", header=None)
    
    # Ensure all columns are int64 to meet DGL's requirements.
    df = df.astype({0: np.int64, 1: np.int64, 2: np.int64, 3: np.int64})
    
    # Shuffle and split the dataset
    train, test, val = np.split(df.sample(frac=1, random_state=1), 
                                  [int(.90 * len(df)), int(.95 * len(df))])
    
    # Write each split to a TSV file in the output directory
    for set_type, data_set in zip(['train', 'test', 'val'], [train, test, val]):
        out_path = os.path.join(output_dir, f"{set_type}.tsv")
        data_set.to_csv(out_path, sep="\t", index=False, header=False)
    
    print("All train, test, and val files produced in", output_dir)

if __name__ == "__main__":
    generate_data()
    train_test_val_split()
