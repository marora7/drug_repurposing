"""
Drug Repurposing Link Prediction Script:
This script performs link prediction for drug repurposing using TransE embeddings
and stores results in a SQLite database using configuration from YAML file.
"""

import os
import numpy as np
import argparse
import pandas as pd
import csv
import sqlite3
import re
from pathlib import Path
from typing import List, Dict, Tuple, Set
import time
from collections import defaultdict
import yaml

def load_config(config_path: str) -> Dict:
    """
    Config loader from YAML.
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_embeddings(entity_path: str, relation_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads the entity and relation embedding matrices from .npy files.
    """
    if not os.path.exists(entity_path):
        raise FileNotFoundError(f"Entity embedding file not found: {entity_path}")
    if not os.path.exists(relation_path):
        raise FileNotFoundError(f"Relation embedding file not found: {relation_path}")
    
    print(f"Loading entity embeddings from {entity_path}")
    entity_embeddings = np.load(entity_path)
    print(f"Loading relation embeddings from {relation_path}")
    relation_embeddings = np.load(relation_path)
    return entity_embeddings, relation_embeddings

def load_entities(entities_file: str) -> Tuple[List[str], Dict[str, int]]:
    """
    Loads entities from the entities.tsv file and creates an ordered list and a lookup dictionary.
    """
    entities = []
    entity_to_idx = {}
    
    print(f"Loading entities from {entities_file}")
    with open(entities_file, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            if len(row) < 2:
                continue
            idx, entity = row[0], row[1]
            entities.append(entity)
            entity_to_idx[entity] = int(idx)
    
    print(f"Loaded {len(entities)} entities")
    return entities, entity_to_idx

def load_relations(relations_file: str) -> Tuple[List[str], Dict[str, int]]:
    """
    Loads relations from the relations.tsv file and creates an ordered list and a lookup dictionary.
    """
    relations = []
    relation_to_idx = {}
    
    print(f"Loading relations from {relations_file}")
    with open(relations_file, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            if len(row) < 2:
                continue
            idx, relation = row[0], row[1]
            relations.append(relation)
            relation_to_idx[relation] = int(idx)
    
    print(f"Loaded {len(relations)} relations")
    return relations, relation_to_idx

def load_known_treatment_triples(train_file: str, disease_entities: List[str], 
                                 treat_relations: List[str]) -> Dict[Tuple[str, str], Set[str]]:
    """
    Efficiently loads only the treatment triples relevant to the specified diseases.
    Returns a dictionary mapping (disease, relation) pairs to sets of known treatment chemicals.
    """
    known_treatments = defaultdict(set)
    disease_set = set(disease_entities)
    relation_set = set(treat_relations)
    
    print(f"Scanning for known treatments in {train_file}")
    line_count = 0
    found_count = 0
    
    # Parse in chunks for memory efficiency
    chunk_size = 100000
    with open(train_file, 'r') as f:
        while True:
            lines = f.readlines(chunk_size)
            if not lines:
                break
                
            for line in lines:
                line_count += 1
                if line_count % 1000000 == 0:
                    print(f"Processed {line_count} triples, found {found_count} relevant treatments")
                
                parts = line.strip().split('\t')
                if len(parts) < 3:
                    continue
                    
                head, relation, tail = parts[0], parts[1], parts[2]
                
                # Note: Direction is (Chemical, treat, Disease)
                if relation in relation_set and tail in disease_set and head.startswith("Chemical::"):
                    known_treatments[(tail, relation)].add(head)  # key is (disease, relation)
                    found_count += 1
    
    print(f"Found {found_count} known treatments after scanning {line_count} triples")
    return known_treatments

def find_disease_entity(disease_name: str, entities: List[str]) -> List[str]:
    """
    Finds disease entities that match the given disease name or ID.
    Handles both partial matches and exact entity IDs.
    """
    # First check if the input is already a complete entity ID in the list
    if disease_name in entities:
        return [disease_name]
    
    # Next check if it's a MESH ID that we can match directly
    if disease_name.startswith("D") and disease_name[1:].isdigit():
        mesh_pattern = f"Disease::Disease:MESH:{disease_name}"
        matched = [entity for entity in entities if mesh_pattern in entity]
        if matched:
            return matched
    
    # Finally try the regex pattern matching
    disease_entities = []
    disease_pattern = re.compile(f"Disease::.*{re.escape(disease_name)}.*", re.IGNORECASE)
    
    for entity in entities:
        if entity.startswith("Disease::") and disease_pattern.match(entity):
            disease_entities.append(entity)
    
    return disease_entities

def find_treat_relations(relations: List[str]) -> List[str]:
    """
    Finds all relations related to treatment.
    """
    treat_relations = []
    for relation in relations:
        # Match relations containing 'treat' between Chemical and Disease
        if 'treat' in relation.lower() and 'chemical:disease' in relation.lower():
            treat_relations.append(relation)
    
    return treat_relations

def filter_chemical_entities(entities: List[str]) -> List[int]:
    """
    Returns indices of chemical entities for efficient filtering.
    """
    chemical_indices = []
    for i, entity in enumerate(entities):
        if entity.startswith("Chemical::"):
            chemical_indices.append(i)
    return chemical_indices

def batch_compute_distances_inverse_direction(tail_idx: int, relation_idx: int, 
                                             entity_emb: np.ndarray, relation_emb: np.ndarray, 
                                             batch_size: int = 10000) -> np.ndarray:
    """
    Computes distances for inverse triple direction (given tail and relation, predict head).
    For TransE, we need h + r ≈ t, so h ≈ t - r
    Therefore we compute distances between entities and (t - r)
    """
    tail_vec = entity_emb[tail_idx]
    rel_vec = relation_emb[relation_idx]
    
    # For inverse direction: predicted_head = tail - relation
    # Note the subtraction instead of addition here
    pred_head = tail_vec - rel_vec
    
    n_entities = entity_emb.shape[0]
    distances = np.zeros(n_entities)
    
    # Process in batches
    for i in range(0, n_entities, batch_size):
        end_idx = min(i + batch_size, n_entities)
        batch = entity_emb[i:end_idx]
        
        # Compute L1 distance (faster than L2 for large vectors)
        batch_distances = np.sum(np.abs(batch - pred_head), axis=1)
        distances[i:end_idx] = batch_distances
    
    return distances

def setup_sqlite_db(db_path: str, table_name: str) -> None:
    """
    Creates a SQLite database and table if they don't exist.
    """
    # Ensure the directory exists
    db_dir = os.path.dirname(db_path)
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create the table if it doesn't exist
    # Use the table name exactly as specified in config
    try:
        cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS "{table_name}" (
            disease TEXT,
            drug TEXT,
            relation TEXT,
            distance REAL,
            status TEXT,
            source_disease TEXT,
            PRIMARY KEY (disease, drug, relation, source_disease)
        )
        """)
        
        conn.commit()
        print(f"Successfully setup table '{table_name}' in database '{db_path}'")
    except sqlite3.Error as e:
        print(f"SQLite error: {e}")
        print("Attempting fallback with sanitized table name...")
        
        # Fallback to sanitized name if there's an error
        sanitized_name = re.sub(r'[^a-zA-Z0-9_]', '_', table_name)
        cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {sanitized_name} (
            disease TEXT,
            drug TEXT,
            relation TEXT,
            distance REAL,
            status TEXT,
            source_disease TEXT,
            PRIMARY KEY (disease, drug, relation, source_disease)
        )
        """)
        conn.commit()
        print(f"Created table with sanitized name '{sanitized_name}'")
        table_name = sanitized_name
    
    conn.close()
    return table_name

def main():
    start_time = time.time()
    parser = argparse.ArgumentParser(
        description="Direction-aware drug repurposing link prediction for diseases"
    )
    parser.add_argument("disease_name", type=str,
                        help="Input disease name (e.g., 'diabetes')")
    parser.add_argument("--top_k", type=int, default=100,
                        help="Number of top predictions to return (default: 100)")
    parser.add_argument("--batch_size", type=int, default=10000,
                        help="Batch size for processing entity embeddings (default: 10000)")
    args = parser.parse_args()
    
    # Determine the path to config.yaml relative to this file
    CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
    config = load_config(CONFIG_PATH)
    
    # Extract parameters from the configuration
    data_dir = Path(config.get("data_dir", "data/processed/train"))
    entities_file = data_dir / config.get("entities_file", "entities.tsv")
    relations_file = data_dir / config.get("relations_file", "relations.tsv")
    train_file = data_dir / config.get("train_file", "train.tsv")
    entity_embedding_path = config.get("entity_embedding_path")
    relation_embedding_path = config.get("relation_embedding_path")
    top_k = args.top_k
    batch_size = args.batch_size
    
    # Database settings from config file
    db_path = config.get("db_path")
    if not db_path and "ml_model" in config and "db_path" in config["ml_model"]:
        db_path = config["ml_model"]["db_path"]
    
    table_name = config.get("table_name")
    if not table_name and "ml_model" in config and "table_name" in config["ml_model"]:
        table_name = config["ml_model"]["table_name"]
    
    # Fallback to defaults if still not found
    if not db_path:
        db_path = "drug_repurposing_results.db"
        print(f"Warning: No db_path found in config, using default: {db_path}")
    
    if not table_name:
        table_name = "repurposing_predictions"
        print(f"Warning: No table_name found in config, using default: {table_name}")
    
    # Set up the SQLite database
    setup_sqlite_db(db_path, table_name)
    
    # Load entities and relations first (smaller files)
    print("Loading entities and relations...")
    entities, entity_to_idx = load_entities(entities_file)
    relations, relation_to_idx = load_relations(relations_file)
    
    # Find disease entities matching the input name
    disease_entities = find_disease_entity(args.disease_name, entities)
    if not disease_entities:
        print(f"No disease entity found matching '{args.disease_name}'.")
        print("Available disease entities:")
        # Show a sample of disease entities
        sample_size = min(20, sum(1 for e in entities if e.startswith("Disease::")))
        samples = [e for e in entities if e.startswith("Disease::")][:sample_size]
        for entity in samples:
            print(f"  {entity}")
        return
    
    # Find treatment-related relations
    treat_relations = find_treat_relations(relations)
    if not treat_relations:
        print("No treatment relations found in the model.")
        return
    
    print(f"\nFound {len(disease_entities)} disease entities matching '{args.disease_name}':")
    for entity in disease_entities:
        print(f"  {entity}")
    
    print(f"\nFound {len(treat_relations)} treatment relations:")
    for relation in treat_relations:
        print(f"  {relation}")
    
    # Load only relevant known treatments (not all triples)
    known_treatments = load_known_treatment_triples(train_file, disease_entities, treat_relations)
    
    # Pre-compute chemical entity indices for filtering
    print("Identifying chemical entities...")
    chemical_indices = filter_chemical_entities(entities)
    print(f"Found {len(chemical_indices)} chemical entities")
    
    # Load embeddings (potentially large files)
    print("Loading embeddings...")
    entity_emb, relation_emb = load_embeddings(entity_embedding_path, relation_embedding_path)
    
    # For each disease entity and treatment relation, predict potential drugs
    all_predictions = []
    
    print("\nRunning predictions...")
    for disease_entity in disease_entities:
        # Disease is the TAIL entity
        disease_idx = entity_to_idx[disease_entity]
        
        for treat_relation in treat_relations:
            relation_idx = relation_to_idx[treat_relation]
            
            print(f"Processing {disease_entity} with relation {treat_relation}")
            # Get known treatments for this disease-relation pair
            known_drugs = known_treatments.get((disease_entity, treat_relation), set())
            
            # Compute distances with INVERSE direction (tail, relation) → head
            distances = batch_compute_distances_inverse_direction(
                disease_idx, relation_idx, entity_emb, relation_emb, batch_size
            )
            
            # Filter and sort predictions (only consider chemical entities)
            # Sort by ascending distance (lower is better)
            sorted_chemical_indices = sorted(
                chemical_indices, 
                key=lambda idx: distances[idx]
            )
            
            # Collect top predictions
            predictions_for_pair = []
            for idx in sorted_chemical_indices:
                if len(predictions_for_pair) >= top_k:
                    break
                
                chemical_entity = entities[idx]
                is_known = chemical_entity in known_drugs
                status = "existing" if is_known else "predicted"
                relation_display = treat_relation if is_known else f"predicted_{treat_relation}"
                
                predictions_for_pair.append({
                    "disease": disease_entity,
                    "drug": chemical_entity,
                    "relation": relation_display,
                    "distance": float(distances[idx]),
                    "status": status,
                    "source_disease": args.disease_name
                })
            
            all_predictions.extend(predictions_for_pair)
    
    # Sort all predictions by distance (ascending order - lower is better)
    all_predictions.sort(key=lambda x: x["distance"])
    
    # Display results
    print("\nTop drug predictions for treating", args.disease_name)
    print("-" * 80)
    print(f"{'Drug':<40} {'Relation':<30} {'Distance':<10} {'Status'}")
    print("-" * 80)
    
    for pred in all_predictions[:top_k]:
        print(f"{pred['drug']:<40} {pred['relation']:<30} {pred['distance']:.4f}   {pred['status']}")
    
    # Convert predictions to DataFrame for easy database storage
    results_df = pd.DataFrame(all_predictions)
    
    # Write results to SQLite database
    print(f"\nWriting results into SQLite database table '{table_name}' at '{db_path}'...")
    conn = sqlite3.connect(db_path)
    try:
        results_df.to_sql(table_name, conn, if_exists="replace", index=False)
        print("Results written successfully.")
    except sqlite3.Error as e:
        print(f"SQLite error: {e}")
        print("Attempting to write with sanitized table name...")
        sanitized_name = re.sub(r'[^a-zA-Z0-9_]', '_', table_name)
        results_df.to_sql(sanitized_name, conn, if_exists="replace", index=False)
        print(f"Results written to sanitized table name '{sanitized_name}'")
    finally:
        conn.commit()
        conn.close()

if __name__ == "__main__":
    main()