"""
ml_input_generation.py

This script generates machine learning input data by reading nodes and edges from compressed CSV files,
processing them into the required format, and splitting the data into train, test, and validation sets.
All output files are saved in the directory specified in config.yaml.
"""

import os
import csv
import numpy as np
import pandas as pd
from time import time
from tqdm import tqdm
from src.utils.config_utils import load_config  # Import the existing load_config

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

def parse_cui(node_id, label):
    """
    Extracts the CUI part from a node ID by removing the label prefix.
    For example, given node_id "Gene:1" and label "Gene", returns "1".
    """
    prefix = f"{label}:"
    if node_id.startswith(prefix):
        return node_id[len(prefix):]
    return node_id

def generate_data():
    """
    Reads nodes and edges from CSV files, processes the data,
    and writes the output in the required format to train.tsv.
    """
    print("Loading nodes...")
    nodes_df = pd.read_csv(nodes_file, compression="gzip")
    
    # Create a lookup dictionary for node details by their ID
    node_map = {}
    for _, row in nodes_df.iterrows():
        node_map[row["Id:ID"]] = {"label": row[":LABEL"], "name": row["name"]}
    
    print("Loading edges...")
    edges_df = pd.read_csv(edges_file, compression="gzip")
    total_edges = len(edges_df)
    print("Total edges in graph:", total_edges)
    
    start_time = time()
    with open(generated_data_filepath, "w", newline="") as tsvfile:
        csv_writer = csv.writer(tsvfile, delimiter="\t")
        # Process each edge to build the output row
        for _, edge in tqdm(edges_df.iterrows(), total=total_edges, desc="Processing edges"):
            start_id = edge[":START_ID"]
            end_id = edge[":END_ID"]
            relation = edge[":TYPE"]
            score = edge["pmcount:int"]
            
            # Get start node details; if not found, default to the prefix from the ID
            start_node = node_map.get(start_id, {"label": start_id.split(":")[0], "name": ""})
            entity1_label = start_node["label"]
            entity1_cui = parse_cui(start_id, entity1_label)
            
            # Get end node details similarly
            end_node = node_map.get(end_id, {"label": end_id.split(":")[0], "name": ""})
            entity2_label = end_node["label"]
            entity2_cui = parse_cui(end_id, entity2_label)
            
            # Construct the output row
            row = [
                f"{entity1_label}::{entity1_cui}",
                f"ALPHAMELD::{relation}:{entity1_label}:{entity2_label}",
                f"{entity2_label}::{entity2_cui}",
                score,
            ]
            csv_writer.writerow(row)
    
    print("Time taken to generate data: {:.2f} seconds".format(time() - start_time))

def train_test_val_split():
    """
    Splits the generated train.tsv data into train, test, and validation sets
    (90%, 5%, 5% respectively) and writes them into the output directory.
    """
    print("Splitting generated data into train, test, and val files...")
    df = pd.read_csv(generated_data_filepath, sep="\t", header=None)
    
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
