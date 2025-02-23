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

# Input file paths from config.yaml
nodes_file = config["nodes_file"]  # e.g., "data/exports/nodes.csv.gz"
edges_file = config["edges_file"]  # e.g., "data/exports/edges.csv.gz"

# Output directory from config.yaml (e.g., "data/processed/train")
output_dir = config["output_dir"]
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# The generated data file will be stored in the output directory as "train.tsv"
generated_data_filepath = os.path.join(output_dir, "train.tsv")

def generate_data():
    """
    Generates training data by reading nodes and edges CSV files, and writes the output
    in TSV format using original string IDs. It also prints the total number of nodes and edges.
    """
    print("Loading nodes...")
    nodes_df = pd.read_csv(nodes_file, compression="gzip")
    
    # Build a lookup dictionary: key is node id, value is its label
    node_map = {}
    for _, row in nodes_df.iterrows():
        node_map[row["Id:ID"]] = row[":LABEL"]
    
    # Count total nodes in graph
    total_nodes = len(nodes_df)
    print("Total nodes in graph:", total_nodes)
    
    print("Loading edges...")
    edges_df = pd.read_csv(edges_file, compression="gzip")
    total_edges = len(edges_df)
    print("Total edges in graph:", total_edges)

    start_time = time()
    with open(generated_data_filepath, "w", newline="") as tsvfile:
        csv_writer = csv.writer(tsvfile, delimiter="\t")
        for _, edge in tqdm(edges_df.iterrows(), total=total_edges, desc="Processing edges"):
            start_id = edge[":START_ID"]
            end_id = edge[":END_ID"]
            relation = edge[":TYPE"]
            score = edge["pmcount:int"]

            # Retrieve labels from node_map (if not found, default to "Unknown")
            entity1_label = node_map.get(start_id, "Unknown")
            entity2_label = node_map.get(end_id, "Unknown")
            
            # Create string representations without any prefix
            entity1_str = f"{entity1_label}::{start_id}"
            entity2_str = f"{entity2_label}::{end_id}"
            # Build relation string without the "ALPHAMELD::" prefix
            relation_str = f"{relation}:{entity1_label}:{entity2_label}"
            
            csv_writer.writerow([entity1_str, relation_str, entity2_str, score])
    
    print("Time taken to generate data: {:.2f} seconds".format(time() - start_time))

def train_test_val_split():
    """
    Splits the generated data into train, test, and validation sets (90%, 5%, 5% respectively)
    and writes them as TSV files in the output directory.
    """
    print("Splitting generated data into train, test, and val files...")
    df = pd.read_csv(generated_data_filepath, sep="\t", header=None)
    
    # Shuffle and split the dataset
    train, test, val = np.split(df.sample(frac=1, random_state=1), 
                                  [int(.90 * len(df)), int(.95 * len(df))])
    
    # Write each split to a TSV file in the output directory
    train.to_csv(os.path.join(output_dir, "train.tsv"), sep="\t", index=False, header=False)
    test.to_csv(os.path.join(output_dir, "test.tsv"), sep="\t", index=False, header=False)
    val.to_csv(os.path.join(output_dir, "val.tsv"), sep="\t", index=False, header=False)
    
    print("All train, test, and val files produced in", output_dir)

if __name__ == "__main__":
    generate_data()
    train_test_val_split()
