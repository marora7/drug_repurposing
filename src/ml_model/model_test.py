"""
Drug Repurposing Link Prediction Script:
This script performs link prediction for drug repurposing using TransE embeddings
and stores results in a SQLite database using configuration from YAML file.
Supports batch processing for efficient GPU utilization.
"""

import os
import sys
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
import torch

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.utils.config_utils import load_config
from src.utils.data_utils import load_tsv_mappings, load_embeddings
from src.utils.db_utils import setup_sqlite_table

def load_known_treatment_triples(train_file: str, disease_entities: List[str], 
                                 treat_relations: List[str]) -> Dict[Tuple[str, str], Set[str]]:
    """
    Loads only the treatment triples relevant to the specified diseases.
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
                                             batch_size: int = 10000,
                                             use_gpu: bool = False) -> np.ndarray:
    """
    Computes distances for inverse triple direction (given tail and relation, predict head).
    For TransE, we need h + r ≈ t, so h ≈ t - r
    Therefore we compute distances between entities and (t - r)
    
    If use_gpu is True, uses PyTorch with CUDA for computation.
    """
    if use_gpu and torch.cuda.is_available():
        # Move data to GPU
        entity_emb_tensor = torch.tensor(entity_emb, device='cuda')
        tail_vec = entity_emb_tensor[tail_idx]
        rel_vec = torch.tensor(relation_emb[relation_idx], device='cuda')
        
        # For inverse direction: predicted_head = tail - relation
        pred_head = tail_vec - rel_vec
        
        n_entities = entity_emb.shape[0]
        distances = torch.zeros(n_entities, device='cuda')
        
        # Process in batches
        for i in range(0, n_entities, batch_size):
            end_idx = min(i + batch_size, n_entities)
            batch = entity_emb_tensor[i:end_idx]
            
            # Compute L1 distance (faster than L2 for large vectors)
            batch_distances = torch.sum(torch.abs(batch - pred_head), dim=1)
            distances[i:end_idx] = batch_distances
        
        # Move result back to CPU
        return distances.cpu().numpy()
    else:
        # Original CPU implementation
        tail_vec = entity_emb[tail_idx]
        rel_vec = relation_emb[relation_idx]
        
        # For inverse direction: predicted_head = tail - relation
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

def process_disease(disease_name, config, top_k, batch_size, use_gpu=False):
    """
    Process a single disease and return predictions.
    Extracted from main() to support batch processing.
    """
    # Extract parameters from the configuration
    data_dir = Path(config.get("data_dir", "data/processed/train"))
    entities_file = data_dir / config.get("entities_file", "entities.tsv")
    relations_file = data_dir / config.get("relations_file", "relations.tsv")
    train_file = data_dir / config.get("train_file", "train.tsv")
    entity_embedding_path = config.get("entity_embedding_path")
    relation_embedding_path = config.get("relation_embedding_path")
    
    # Get table name from the ml_model section of config
    table_name = config.get("ml_model", {}).get("table_name")
    
    # Database settings from config file
    db_path = config.get("ml_model", {}).get("db_path")
    
    # Define the table schema
    schema = {
        'disease': 'TEXT',
        'drug': 'TEXT', 
        'relation': 'TEXT',
        'distance': 'REAL',
        'status': 'TEXT',
        'source_disease': 'TEXT'
    }
    
    # Set up the table in the existing SQLite database
    setup_sqlite_table(db_path, table_name, schema, "disease, drug, relation, source_disease")
    
    # Load entities and relations
    entities, entity_to_idx = load_tsv_mappings(entities_file, "entities")
    relations, relation_to_idx = load_tsv_mappings(relations_file, "relations")
    
    # Find disease entities matching the input name
    disease_entities = find_disease_entity(disease_name, entities)
    if not disease_entities:
        print(f"No disease entity found matching '{disease_name}'.")
        return []
    
    # Find treatment-related relations
    treat_relations = find_treat_relations(relations)
    if not treat_relations:
        print("No treatment relations found in the model.")
        return []
    
    print(f"Processing disease: {disease_name}")
    print(f"Found {len(disease_entities)} matching entities and {len(treat_relations)} treatment relations")
    
    # Load only relevant known treatments (not all triples)
    known_treatments = load_known_treatment_triples(train_file, disease_entities, treat_relations)
    
    # Pre-compute chemical entity indices for filtering
    chemical_indices = filter_chemical_entities(entities)
    
    # Load embeddings using the imported load_embeddings function
    embedding_paths = [entity_embedding_path, relation_embedding_path]
    embedding_names = ["entity", "relation"]
    entity_emb, relation_emb = load_embeddings(embedding_paths, embedding_names)
    
    # For each disease entity and treatment relation, predict potential drugs
    all_predictions = []
    
    for disease_entity in disease_entities:
        # Disease is the TAIL entity
        disease_idx = entity_to_idx[disease_entity]
        
        for treat_relation in treat_relations:
            relation_idx = relation_to_idx[treat_relation]
            
            # Get known treatments for this disease-relation pair
            known_drugs = known_treatments.get((disease_entity, treat_relation), set())
            
            # Compute distances with INVERSE direction (tail, relation) → head
            distances = batch_compute_distances_inverse_direction(
                disease_idx, relation_idx, entity_emb, relation_emb, batch_size, use_gpu
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
                    "source_disease": disease_name
                })
            
            all_predictions.extend(predictions_for_pair)
    
    # Sort all predictions by distance (ascending order - lower is better)
    all_predictions.sort(key=lambda x: x["distance"])
    
    # Display top 5 results (to keep output manageable)
    print(f"Top 5 drug predictions for treating {disease_name}:")
    for pred in all_predictions[:5]:
        print(f"{pred['drug']} (Distance: {pred['distance']:.4f}, Status: {pred['status']})")
    
    # Convert predictions to DataFrame for easy database storage
    results_df = pd.DataFrame(all_predictions)
    
    # Write results to SQLite database
    conn = sqlite3.connect(db_path)
    try:
        results_df.to_sql(table_name, conn, if_exists="append", index=False)
        print(f"Successfully processed {disease_name} with {len(all_predictions)} predictions")
    except sqlite3.Error as e:
        print(f"SQLite error for {disease_name}: {e}")
    finally:
        conn.commit()
        conn.close()
    
    return all_predictions

def main():
    start_time = time.time()

    parser = argparse.ArgumentParser(
        description="Drug repurposing link prediction for diseases"
    )
    
    # Add arguments for both single disease and batch processing
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("disease_name", type=str, nargs='?', default=None,
                      help="Input disease name (e.g., 'diabetes')")
    group.add_argument("--batch_file", type=str, 
                      help="Path to file containing disease names/IDs (one per line)")
    
    # Other parameters
    parser.add_argument("--top_k", type=int, default=100,
                        help="Number of top predictions to return (default: 100)")
    parser.add_argument("--batch_size", type=int, default=10000,
                        help="Batch size for processing entity embeddings (default: 10000)")
    parser.add_argument("--gpu", type=int, default=None,
                        help="GPU device index to use (omit for CPU-only)")
    args = parser.parse_args()
    
    # Set the GPU device if specified
    use_gpu = args.gpu is not None
    if use_gpu:
        if not torch.cuda.is_available():
            print("Warning: GPU requested but PyTorch CUDA is not available. Falling back to CPU.")
            use_gpu = False
        else:
            try:
                gpu_device = args.gpu
                torch.cuda.set_device(gpu_device)
                print(f"Using GPU device: {torch.cuda.get_device_name(gpu_device)}")
                
                # Print memory info
                total_mem = torch.cuda.get_device_properties(gpu_device).total_memory / 1e9
                allocated_mem = torch.cuda.memory_allocated(gpu_device) / 1e9
                print(f"GPU memory: {allocated_mem:.2f} GB allocated / {total_mem:.2f} GB total")
            except Exception as e:
                print(f"Error setting GPU device: {e}")
                print("Falling back to CPU.")
                use_gpu = False
    
    # Determine the path to config.yaml relative to this file
    CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
    config = load_config(CONFIG_PATH)
    
    # Process either single disease or batch file
    if args.batch_file:
        # Process in batch mode
        if not os.path.exists(args.batch_file):
            print(f"Error: Batch file {args.batch_file} not found.")
            return
        
        print(f"Processing diseases in batch from file: {args.batch_file}")
        with open(args.batch_file, 'r') as f:
            disease_names = [line.strip() for line in f if line.strip()]
        
        print(f"Found {len(disease_names)} diseases to process")
        
        # Track overall statistics
        successful = 0
        failed = 0
        
        # Process each disease
        for i, disease_name in enumerate(disease_names):
            try:
                print(f"\nProcessing [{i+1}/{len(disease_names)}]: {disease_name}")
                predictions = process_disease(
                    disease_name, config, args.top_k, args.batch_size, use_gpu
                )
                
                if predictions:
                    successful += 1
                    print(f"Successfully processed {disease_name}")
                else:
                    failed += 1
                    print(f"Failed to process {disease_name} (no predictions found)")
            except Exception as e:
                failed += 1
                print(f"Error processing {disease_name}: {str(e)}")
        
        # Print batch summary
        print("\n" + "="*50)
        print(f"Batch processing completed: {successful} successful, {failed} failed")
        print(f"Total execution time: {time.time() - start_time:.2f} seconds")
        print("="*50)
        
    else:
        # Process single disease (original behavior)
        disease_name = args.disease_name
        print(f"Processing single disease: {disease_name}")
        
        try:
            predictions = process_disease(
                disease_name, config, args.top_k, args.batch_size, use_gpu
            )
            
            # Display detailed results for single disease mode
            if predictions:
                print("\nTop drug predictions for treating", disease_name)
                print("-" * 80)
                print(f"{'Drug':<40} {'Relation':<30} {'Distance':<10} {'Status'}")
                print("-" * 80)
                
                for pred in predictions[:args.top_k]:
                    print(f"{pred['drug']:<40} {pred['relation']:<30} {pred['distance']:.4f}   {pred['status']}")
                
                print(f"\nTotal execution time: {time.time() - start_time:.2f} seconds")
            else:
                print(f"No predictions found for {disease_name}")
        except Exception as e:
            print(f"Error processing {disease_name}: {str(e)}")
    
    # Clean up GPU resources if used
    if use_gpu and torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()