"""
Disease Prediction Orchestrator:
This script manages the execution of drug repurposing predictions for multiple disease entities.
It handles parallel processing, checkpointing, and monitors progress across all disease entities.
"""

import os
import sys
import argparse
import sqlite3
import subprocess
import multiprocessing
import logging
import time
import pandas as pd
import yaml
from pathlib import Path
from typing import List, Dict, Set, Tuple
import csv
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("disease_orchestrator.log"),
        logging.StreamHandler()
    ]
)

def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def setup_database(db_path: str) -> None:
    """
    Sets up the database with necessary tables for tracking progress.
    """
    # Ensure the directory exists
    db_dir = os.path.dirname(db_path)
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir)
        
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create a table to track processed diseases
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS processed_diseases (
        disease_id TEXT PRIMARY KEY,
        processed_at TIMESTAMP,
        status TEXT,
        processing_time REAL
    )
    """)
    
    # Create a table to store overall progress
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS orchestrator_progress (
        id INTEGER PRIMARY KEY,
        total_diseases INTEGER,
        processed_diseases INTEGER,
        start_time TIMESTAMP,
        last_update TIMESTAMP,
        estimated_completion TIMESTAMP
    )
    """)
    
    conn.commit()
    conn.close()
    
def load_all_disease_entities(entities_file: str) -> List[str]:
    """
    Extracts all disease entities from the entities file.
    """
    diseases = []
    
    with open(entities_file, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            if len(row) < 2:
                continue
            entity = row[1]
            if entity.startswith("Disease::"):
                diseases.append(entity)
    
    return diseases

def get_processed_diseases(db_path: str) -> Set[str]:
    """
    Retrieves the set of disease entities that have already been processed.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT disease_id FROM processed_diseases WHERE status = 'completed'")
    processed = {row[0] for row in cursor.fetchall()}
    
    conn.close()
    return processed

def update_progress(db_path: str, total_diseases: int, processed_count: int) -> None:
    """
    Updates the progress tracking information.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    now = datetime.now()
    
    # Check if we have an existing progress record
    cursor.execute("SELECT id, start_time FROM orchestrator_progress")
    progress_record = cursor.fetchone()
    
    if progress_record:
        progress_id, start_time_str = progress_record
        start_time = datetime.fromisoformat(start_time_str)
        
        # Calculate estimated completion time
        if processed_count > 0:
            elapsed = (now - start_time).total_seconds()
            seconds_per_disease = elapsed / processed_count
            remaining_diseases = total_diseases - processed_count
            seconds_remaining = seconds_per_disease * remaining_diseases
            estimated_completion = now + timedelta(seconds=seconds_remaining)
        else:
            estimated_completion = now  # Default if no progress yet
        
        # Update existing record
        cursor.execute("""
        UPDATE orchestrator_progress 
        SET processed_diseases = ?, last_update = ?, estimated_completion = ?
        WHERE id = ?
        """, (processed_count, now.isoformat(), estimated_completion.isoformat(), progress_id))
    else:
        # Create new progress record
        cursor.execute("""
        INSERT INTO orchestrator_progress
        (total_diseases, processed_diseases, start_time, last_update, estimated_completion)
        VALUES (?, ?, ?, ?, ?)
        """, (total_diseases, processed_count, now.isoformat(), now.isoformat(), now.isoformat()))
    
    conn.commit()
    conn.close()

def mark_disease_started(db_path: str, disease_id: str) -> None:
    """
    Marks a disease as being processed.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
    INSERT OR REPLACE INTO processed_diseases
    (disease_id, processed_at, status, processing_time)
    VALUES (?, ?, ?, ?)
    """, (disease_id, datetime.now().isoformat(), 'in_progress', 0))
    
    conn.commit()
    conn.close()

def mark_disease_completed(db_path: str, disease_id: str, processing_time: float) -> None:
    """
    Marks a disease as successfully processed.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
    UPDATE processed_diseases
    SET status = ?, processed_at = ?, processing_time = ?
    WHERE disease_id = ?
    """, ('completed', datetime.now().isoformat(), processing_time, disease_id))
    
    conn.commit()
    conn.close()

def mark_disease_failed(db_path: str, disease_id: str) -> None:
    """
    Marks a disease as failed during processing.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
    UPDATE processed_diseases
    SET status = ?, processed_at = ?
    WHERE disease_id = ?
    """, ('failed', datetime.now().isoformat(), disease_id))
    
    conn.commit()
    conn.close()

def process_disease(disease_id: str, config_path: str) -> bool:
    """
    Processes a single disease entity by calling the prediction script.
    Returns True if successful, False otherwise.
    """
    start_time = time.time()
    
    try:
        config = load_config(config_path)
        db_path = config.get("ml_model", {}).get("db_path", "drug_repurposing_results.db")
        
        # Mark disease as being processed
        mark_disease_started(db_path, disease_id)
        
        # Prepare command to run the model_test.py script
        script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_test.py")
        
        # Run the prediction script for this disease
        logging.info(f"Processing disease: {disease_id}")
        result = subprocess.run(
            ["python", script_path, disease_id],
            capture_output=True,
            text=True,
            check=True
        )
        
        processing_time = time.time() - start_time
        mark_disease_completed(db_path, disease_id, processing_time)
        
        logging.info(f"Completed disease {disease_id} in {processing_time:.2f} seconds")
        return True
        
    except Exception as e:
        processing_time = time.time() - start_time
        logging.error(f"Error processing disease {disease_id}: {str(e)}")
        mark_disease_failed(db_path, disease_id)
        return False

def worker_function(args):
    """
    Worker function for multiprocessing.
    """
    disease_id, config_path = args
    return process_disease(disease_id, config_path)

def main():
    parser = argparse.ArgumentParser(
        description="Orchestrate drug repurposing predictions for multiple disease entities"
    )
    parser.add_argument("--config", type=str, default=None,
                        help="Path to configuration file")
    parser.add_argument("--processes", type=int, default=4,
                        help="Number of parallel processes to use (default: 4)")
    parser.add_argument("--batch-size", type=int, default=100,
                        help="Number of diseases to process in each batch (default: 100)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from previous run")
    args = parser.parse_args()
    
    # Determine config path
    if args.config:
        config_path = args.config
    else:
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'config', 'config.yaml')
    
    if not os.path.exists(config_path):
        logging.error(f"Config file not found at {config_path}")
        sys.exit(1)
    
    # Load configuration
    config = load_config(config_path)
    
    # Extract database information
    db_path = config.get("ml_model", {}).get("db_path", "drug_repurposing_results.db")
    
    # Extract path to entities file
    data_dir = Path(config.get("data_dir", "data/processed/train"))
    entities_file = data_dir / config.get("entities_file", "entities.tsv")
    
    # Setup tracking database
    setup_database(db_path)
    
    # Load all disease entities
    logging.info(f"Loading disease entities from {entities_file}")
    all_diseases = load_all_disease_entities(str(entities_file))
    logging.info(f"Found {len(all_diseases)} disease entities")
    
    # Get already processed diseases if resuming
    processed_diseases = set()
    if args.resume:
        processed_diseases = get_processed_diseases(db_path)
        logging.info(f"Resuming from previous run. {len(processed_diseases)} diseases already processed")
    
    # Filter out already processed diseases
    remaining_diseases = [d for d in all_diseases if d not in processed_diseases]
    logging.info(f"{len(remaining_diseases)} diseases remaining to process")
    
    # Initialize progress tracking
    update_progress(db_path, len(all_diseases), len(processed_diseases))
    
    # Process in batches
    batch_size = args.batch_size
    processes = min(args.processes, multiprocessing.cpu_count())
    
    logging.info(f"Using {processes} parallel processes with batch size {batch_size}")
    
    total_processed = len(processed_diseases)
    total_batches = (len(remaining_diseases) + batch_size - 1) // batch_size
    
    for batch_num, batch_start in enumerate(range(0, len(remaining_diseases), batch_size)):
        batch_end = min(batch_start + batch_size, len(remaining_diseases))
        batch = remaining_diseases[batch_start:batch_end]
        
        logging.info(f"Processing batch {batch_num+1}/{total_batches} ({len(batch)} diseases)")
        
        # Prepare arguments for each disease in the batch
        args_list = [(disease, config_path) for disease in batch]
        
        # Process the batch in parallel
        with multiprocessing.Pool(processes=processes) as pool:
            results = pool.map(worker_function, args_list)
        
        # Update progress
        successful = sum(1 for r in results if r)
        total_processed += successful
        
        update_progress(db_path, len(all_diseases), total_processed)
        
        logging.info(f"Batch complete. {successful}/{len(batch)} successful. "
                    f"Total progress: {total_processed}/{len(all_diseases)} "
                    f"({total_processed/len(all_diseases)*100:.1f}%)")
    
    logging.info("All disease processing complete!")

if __name__ == "__main__":
    main()
