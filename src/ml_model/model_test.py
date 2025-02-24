"""
Modified Link Prediction Script for Drug Repurposing:

This script performs two main tasks for each input disease:
1. Generates a benchmark set of drug–disease pairs from the existing "treat" relationships.
2. Generates a candidate set of drug–disease pairs by pairing every chemical (drug) with the disease,
   excluding those that already have a "treat" relationship.
For each pair, it computes the predicted distance as:
    distance = || (drug_embedding + treat_embedding) - disease_embedding ||
Benchmark rows keep the relation as "treat" while candidate rows are labeled as "predicted_treat".
Finally, the combined results are written to the SQLite table specified in the configuration.
"""

import os
import numpy as np
import argparse
import pandas as pd
import torch
import sqlite3
from src.utils.config_utils import load_config

def load_embeddings(entity_path, relation_path, device):
    """
    Loads entity and relation embeddings from .npy files,
    converts them to PyTorch tensors, and moves them to the specified device.
    """
    if not os.path.exists(entity_path):
        raise FileNotFoundError(f"Entity embedding file not found: {entity_path}")
    if not os.path.exists(relation_path):
        raise FileNotFoundError(f"Relation embedding file not found: {relation_path}")
    
    entity_embeddings = torch.from_numpy(np.load(entity_path)).float().to(device)
    relation_embeddings = torch.from_numpy(np.load(relation_path)).float().to(device)
    return entity_embeddings, relation_embeddings

def load_ml_model_df(nodes_csv, edges_csv, ml_query):
    """
    Loads nodes and edges CSV files (with optional gzip compression),
    creates an in-memory SQLite database, and executes the ml_model SQL query.
    Returns the resulting DataFrame.
    """
    comp_nodes = 'gzip' if nodes_csv.endswith('.gz') else None
    comp_edges = 'gzip' if edges_csv.endswith('.gz') else None

    nodes_df = pd.read_csv(nodes_csv, compression=comp_nodes)
    edges_df = pd.read_csv(edges_csv, compression=comp_edges)
    
    conn = sqlite3.connect(":memory:")
    nodes_df.to_sql("nodes", conn, index=False, if_exists="replace")
    edges_df.to_sql("edges", conn, index=False, if_exists="replace")
    ml_model_df = pd.read_sql_query(ml_query, conn)
    conn.close()
    return ml_model_df

def load_entity_mapping(nodes_csv):
    """
    Loads the nodes CSV file and creates a mapping from "Id:ID" to its row index.
    Assumes that the ordering in the CSV corresponds to the ordering of the entity embeddings.
    """
    comp = 'gzip' if nodes_csv.endswith('.gz') else None
    df = pd.read_csv(nodes_csv, compression=comp)
    mapping = {row["Id:ID"]: i for i, row in df.iterrows()}
    return mapping

def load_relation_mapping(train_file):
    """
    Reads the training TSV file and returns a dictionary mapping each unique relation
    (based on its first appearance) to an index.
    """
    relations = []
    seen = set()
    with open(train_file, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            relation = parts[1]
            if relation not in seen:
                seen.add(relation)
                relations.append(relation)
    mapping = {rel: idx for idx, rel in enumerate(relations)}
    return mapping

def generate_benchmark_predictions(disease, ml_model_df, entity_mapping, entity_emb, relation_mapping, relation_emb):
    """
    Generates the benchmark set by filtering ml_model_df for rows where:
      - disease_name matches the input disease, and
      - an existing "treat" relationship exists (drug_to_disease == True).
    For each pair, computes the distance using the treat embedding.
    """
    # Use case-insensitive matching for disease
    df_bench = ml_model_df[ml_model_df['disease_name'].str.lower() == disease.lower()]
    df_bench = df_bench[df_bench['drug_to_disease'] == True]
    results = []
    for _, row in df_bench.iterrows():
        drug_id = row["drug_id"]
        dis_id = row["disease_id"]
        if drug_id not in entity_mapping or dis_id not in entity_mapping:
            continue
        drug_idx = entity_mapping[drug_id]
        dis_idx = entity_mapping[dis_id]
        drug_vec = entity_emb[drug_idx]
        dis_vec = entity_emb[dis_idx]
        # Use the treat relation embedding from the training mapping
        if "treat" in relation_mapping:
            rel_idx = relation_mapping["treat"]
            rel_vec = relation_emb[rel_idx]
        else:
            rel_vec = torch.zeros_like(drug_vec)
        predicted_tail = drug_vec + rel_vec
        distance = torch.norm(predicted_tail - dis_vec).item()
        results.append({
            "disease": disease,
            "drug_id": drug_id,
            "drug_name": row["drug_name"],
            "disease_id": dis_id,
            "disease_name": row["disease_name"],
            "relation": "treat",
            "pmid": row["pmid"],
            "drug_to_disease": True,
            "distance": distance
        })
    return pd.DataFrame(results)

def generate_candidate_predictions(disease, nodes_csv, benchmark_drug_ids, entity_mapping, entity_emb, relation_mapping, relation_emb):
    """
    Generates candidate drug–disease pairs by taking all chemicals (from nodes CSV)
    and pairing them with the given disease, excluding those drugs already present in benchmark_drug_ids.
    For each candidate pair, computes the predicted distance using the treat embedding.
    Labels the relation as "predicted_treat" and sets drug_to_disease to False.
    """
    comp = 'gzip' if nodes_csv.endswith('.gz') else None
    nodes_df = pd.read_csv(nodes_csv, compression=comp)
    # Filter for chemicals
    chem_df = nodes_df[nodes_df[":LABEL"] == "Chemical"]
    # Get the disease row (assumes disease name matching on 'name' column, case-insensitive)
    disease_df = nodes_df[(nodes_df[":LABEL"] == "Disease") & (nodes_df["name"].str.lower() == disease.lower())]
    if disease_df.empty:
        print(f"No disease found for {disease}")
        return pd.DataFrame()
    disease_row = disease_df.iloc[0]
    disease_id = disease_row["Id:ID"]
    disease_name = disease_row["name"]

    results = []
    for _, row in chem_df.iterrows():
        drug_id = row["Id:ID"]
        # Skip drugs already in the benchmark for this disease
        if drug_id in benchmark_drug_ids:
            continue
        if drug_id not in entity_mapping or disease_id not in entity_mapping:
            continue
        drug_idx = entity_mapping[drug_id]
        dis_idx = entity_mapping[disease_id]
        drug_vec = entity_emb[drug_idx]
        dis_vec = entity_emb[dis_idx]
        if "treat" in relation_mapping:
            rel_idx = relation_mapping["treat"]
            rel_vec = relation_emb[rel_idx]
        else:
            rel_vec = torch.zeros_like(drug_vec)
        predicted_tail = drug_vec + rel_vec
        distance = torch.norm(predicted_tail - dis_vec).item()
        results.append({
            "disease": disease,
            "drug_id": drug_id,
            "drug_name": row["name"],
            "disease_id": disease_id,
            "disease_name": disease_name,
            "relation": "predicted_treat",
            "pmid": 0,  # No pmid for candidate predictions
            "drug_to_disease": False,
            "distance": distance
        })
    return pd.DataFrame(results)

def main():
    parser = argparse.ArgumentParser(
        description="Link prediction for drug repurposing using benchmark and candidate generation."
    )
    parser.add_argument("disease_names", type=str,
                        help="Comma-separated list of disease names (e.g., 'Diabetes,Hypertension')")
    args = parser.parse_args()
    
    # Determine the path to config.yaml relative to this file (assumes config.yaml is in project_root/config/)
    CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
    config = load_config(CONFIG_PATH)
    
    # Extract configuration parameters
    nodes_file = config["transformation"]["nodes"]["output"]   # e.g., data/exports/nodes.csv.gz
    edges_file = config["transformation"]["edges"]["output"]     # e.g., data/exports/edges.csv.gz
    ml_query = config["ml_model"]["query"]
    train_file = config["train_file"]
    entity_embedding_path = config["entity_embedding_path"]
    relation_embedding_path = config["relation_embedding_path"]
    # Output info now resides under ml_model.output in the YAML
    output_db_path = config["ml_model"]["output"]["db_path"]
    output_table = config["ml_model"]["output"]["table_name"]
    
    # Set up device (CUDA if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load embeddings
    print("Loading embeddings...")
    entity_emb, relation_emb = load_embeddings(entity_embedding_path, relation_embedding_path, device)
    
    # Load the ml_model DataFrame (for benchmark extraction)
    print("Loading ml_model DataFrame using SQL query...")
    ml_model_df = load_ml_model_df(nodes_file, edges_file, ml_query)
    
    # Create entity and relation mappings
    print("Loading entity mapping...")
    entity_mapping = load_entity_mapping(nodes_file)
    print("Loading relation mapping...")
    relation_mapping = load_relation_mapping(train_file)
    
    all_results = []
    # Process each disease in the input (comma-separated)
    diseases = [d.strip() for d in args.disease_names.split(",")]
    for disease in diseases:
        print(f"\nProcessing disease: {disease}")
        # Generate benchmark predictions from existing "treat" relationships
        benchmark_df = generate_benchmark_predictions(disease, ml_model_df, entity_mapping, entity_emb, relation_mapping, relation_emb)
        # Extract drug IDs from benchmark to exclude them from candidates
        benchmark_drug_ids = set(benchmark_df["drug_id"].tolist())
        # Generate candidate predictions by pairing all chemicals with the disease (excluding benchmark drugs)
        candidate_df = generate_candidate_predictions(disease, nodes_file, benchmark_drug_ids, entity_mapping, entity_emb, relation_mapping, relation_emb)
        # Combine benchmark and candidate predictions
        disease_results = pd.concat([benchmark_df, candidate_df], ignore_index=True)
        all_results.append(disease_results)
    
    # Combine results for all diseases
    if all_results:
        results_df = pd.concat(all_results, ignore_index=True)
    else:
        results_df = pd.DataFrame()
    
    if results_df.empty:
        print("No prediction results found.")
    else:
        print("\nCombined Prediction Results:")
        print(results_df.to_string(index=False))
    
    # Write the complete results into the output SQLite table
    print(f"\nWriting results into SQLite database table '{output_table}' at '{output_db_path}'...")
    conn = sqlite3.connect(output_db_path)
    results_df.to_sql(output_table, conn, if_exists="replace", index=False)
    conn.commit()
    conn.close()
    print("Results written successfully.")

if __name__ == "__main__":
    main()

