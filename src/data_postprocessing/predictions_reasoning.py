"""
Evidence extraction for drug repurposing predictions.
This script extracts drugs and diseases with 'predicted_treat' relations,
runs them through an LLM for mechanism analysis, searches for web evidence,
and combines both sources of information.
"""

import os
import sys
import time
import json
import sqlite3
import logging
import argparse
import datetime
import requests
import threading
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from pydantic import BaseModel, Field

# Get the absolute path of the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the project root directory (assuming the script is in src/data_postprocessing/)
project_root = os.path.abspath(os.path.join(current_dir, '../..'))
# Add the project root to sys.path
sys.path.insert(0, project_root)
from src.utils.config_utils import load_config

# Set up argument parser
parser = argparse.ArgumentParser(description='Extract evidence for drug repurposing predictions')
parser.add_argument('--config-path', type=str, default='config.yaml', help='Path to the config file')
parser.add_argument('--batch-size', type=int, default=50, help='Batch size for processing')
parser.add_argument('--max-pairs', type=int, default=None, help='Maximum number of drug-disease pairs to process')
parser.add_argument('--ollama-url', type=str, default='http://localhost:11434', help='URL for Ollama API')
parser.add_argument('--serper-api-key', type=str, help='API key for Serper')
parser.add_argument('--workers', type=int, default=5, help='Number of worker threads')
args = parser.parse_args()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Concurrency control
db_lock = threading.Lock()
api_semaphore = threading.Semaphore(3)  # Limit concurrent API calls

# Pydantic models for data validation and structure
class DrugDiseasePair(BaseModel):
    drug_id: str
    drug_name: str
    drug_category: str
    disease_id: str
    disease_name: str
    disease_type: str
    therapeutic_area: str
    distance: float

class DrugDiseaseEvidence(BaseModel):
    drug_id: str
    drug_name: str
    disease_id: str
    disease_name: str
    llm_explanation: str
    web_explanation: str
    web_links: List[str]
    combined_confidence: str
    source: str
    timestamp: str

def get_db_connection(db_path: str) -> sqlite3.Connection:
    """Create a connection to the SQLite database."""
    conn = sqlite3.connect(db_path, timeout=60)  # Increase timeout for locks
    conn.row_factory = sqlite3.Row
    return conn

def setup_database(db_path: str) -> None:
    """Set up the database tables and indexes."""
    with db_lock:
        conn = get_db_connection(db_path)
        try:
            cursor = conn.cursor()
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions_reasoning (
                drug_id TEXT,
                drug_name TEXT,
                disease_id TEXT,
                disease_name TEXT,
                llm_explanation TEXT,
                web_explanation TEXT,
                web_links TEXT,
                combined_confidence TEXT,
                source TEXT,
                timestamp TEXT,
                PRIMARY KEY (drug_id, disease_id)
            )
            ''')
            # Create index for faster lookups
            cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_drug_disease 
            ON predictions_reasoning(drug_id, disease_id)
            ''')
            conn.commit()
        finally:
            conn.close()

def fetch_unprocessed_pairs(db_path: str, limit: Optional[int] = None) -> List[DrugDiseasePair]:
    """Fetch drug-disease pairs that haven't been processed yet."""
    with db_lock:
        conn = get_db_connection(db_path)
        try:
            # First, get all pairs from the source table
            query = """
            SELECT 
                r.disease_id, d.disease_name, d.disease_type_name, d.therapeutic_area_name,
                r.drug_id, dr.drug_name, dr.category_name, r.distance
            FROM 
                relations_after_prediction r
            JOIN 
                diseases_grouped d ON r.disease_id = d.disease_id
            JOIN 
                drugs_grouped dr ON r.drug_id = dr.drug_id
            WHERE 
                r.relation = 'predicted_treat'
                AND dr.category_name = 'Approved Drugs'
                AND d.disease_type_name = 'Primary Treatable Diseases'
                AND r.distance <= 150
            ORDER BY r.distance ASC
            """
            
            if limit is not None:
                query += f" LIMIT {limit}"
                
            cursor = conn.cursor()
            cursor.execute(query)
            rows = cursor.fetchall()
            
            # Get all processed pairs
            cursor.execute("""
            SELECT drug_id, disease_id FROM predictions_reasoning
            """)
            processed = set((row['drug_id'], row['disease_id']) for row in cursor.fetchall())
            
            # Create list of unprocessed pairs
            pairs = []
            for row in rows:
                pair_key = (row['drug_id'], row['disease_id'])
                if pair_key not in processed:
                    pair = DrugDiseasePair(
                        disease_id=row['disease_id'],
                        disease_name=row['disease_name'],
                        disease_type=row['disease_type_name'],
                        therapeutic_area=row['therapeutic_area_name'],
                        drug_id=row['drug_id'],
                        drug_name=row['drug_name'],
                        drug_category=row['category_name'],
                        distance=row['distance']
                    )
                    pairs.append(pair)
            
            logger.info(f"Found {len(pairs)} unprocessed pairs out of {len(rows)} total pairs")
            return pairs
        finally:
            conn.close()

def combined_analysis(pair: DrugDiseasePair, ollama_url: str, serper_api_key: str) -> DrugDiseaseEvidence:
    """Run combined analysis for a single drug-disease pair."""
    timestamp = datetime.datetime.now().isoformat()
    
    try:
        # Limit concurrent API calls
        with api_semaphore:
            # Get combined explanation and search queries in a single LLM call
            prompt = f"""
            You are a biomedical expert analyzing the potential repurposing of {pair.drug_name} to treat {pair.disease_name}.
            
            Your response should have EXACTLY two parts separated by "===SEARCH QUERIES===".
            
            PART 1: Provide a brief explanation (150 words max) of:
            - The mechanism of action of {pair.drug_name}
            - The pathophysiology of {pair.disease_name}
            - How the drug might address the disease
            
            ===SEARCH QUERIES===
            
            PART 2: Provide exactly 1 search query to find scientific evidence about this potential treatment.
            
            DO NOT include any other text, just the explanation followed by the separator and then the query.
            """
            
            try:
                response = requests.post(
                    f"{ollama_url}/api/generate",
                    json={
                        "model": "deepseek-r1",
                        "prompt": prompt,
                        "stream": False
                    },
                    timeout=60  # 60 second timeout
                )
                
                if response.status_code != 200:
                    logger.error(f"Failed LLM call: {response.text}")
                    llm_explanation = "Failed to generate explanation"
                    search_query = f"{pair.drug_name} {pair.disease_name} treatment"
                else:
                    result_text = response.json().get("response", "")
                    parts = result_text.split("===SEARCH QUERIES===")
                    
                    if len(parts) >= 2:
                        llm_explanation = parts[0].strip()
                        search_query = parts[1].strip().split('\n')[0].strip()
                    else:
                        llm_explanation = result_text
                        search_query = f"{pair.drug_name} {pair.disease_name} treatment"
            except Exception as e:
                logger.error(f"LLM API error: {str(e)}")
                llm_explanation = f"Error: {str(e)}"
                search_query = f"{pair.drug_name} {pair.disease_name} treatment"
        
        # Web search with single query
        with api_semaphore:
            web_links = []
            web_text = ""
            
            try:
                url = "https://google.serper.dev/search"
                payload = json.dumps({
                    "q": search_query,
                    "gl": "us",
                    "hl": "en",
                    "num": 3  # Limit to 3 results
                })
                headers = {
                    'X-API-KEY': serper_api_key,
                    'Content-Type': 'application/json'
                }
                
                response = requests.post(url, headers=headers, data=payload, timeout=30)
                
                if response.status_code == 200:
                    results = response.json()
                    
                    for item in results.get("organic", [])[:2]:  # Just use top 2 results
                        title = item.get("title", "")
                        snippet = item.get("snippet", "")
                        link = item.get("link", "")
                        
                        if link:
                            web_links.append(link)
                            web_text += f"Title: {title}\nSnippet: {snippet}\n\n"
            except Exception as e:
                logger.error(f"Web search error: {str(e)}")
        
        # Simplified web explanation
        if web_text:
            web_explanation = f"Web results for {pair.drug_name} treating {pair.disease_name}:\n\n{web_text[:500]}..."
        else:
            web_explanation = "No relevant information found on the web."
        
        # Simple confidence determination
        if len(llm_explanation) > 30 and "failed" not in llm_explanation.lower() and web_links:
            combined_confidence = "Medium"
        elif len(llm_explanation) > 30 and "failed" not in llm_explanation.lower():
            combined_confidence = "Low"
        else:
            combined_confidence = "Low"
        
        # Source determination
        has_llm = len(llm_explanation) > 30 and "failed" not in llm_explanation.lower()
        has_web = len(web_links) > 0
        
        if has_llm and has_web:
            source = "Both"
        elif has_llm:
            source = "LLM"
        elif has_web:
            source = "Web"
        else:
            source = "Unknown"
        
        return DrugDiseaseEvidence(
            drug_id=pair.drug_id,
            drug_name=pair.drug_name,
            disease_id=pair.disease_id,
            disease_name=pair.disease_name,
            llm_explanation=llm_explanation,
            web_explanation=web_explanation,
            web_links=web_links,
            combined_confidence=combined_confidence,
            source=source,
            timestamp=timestamp
        )
    except Exception as e:
        logger.error(f"Error processing {pair.drug_name} for {pair.disease_name}: {str(e)}")
        return DrugDiseaseEvidence(
            drug_id=pair.drug_id,
            drug_name=pair.drug_name,
            disease_id=pair.disease_id,
            disease_name=pair.disease_name,
            llm_explanation=f"Error: {str(e)}",
            web_explanation="Error during processing",
            web_links=[],
            combined_confidence="Low",
            source="Error",
            timestamp=timestamp
        )

def save_evidence(db_path: str, evidence: DrugDiseaseEvidence) -> None:
    """Save a single evidence entry to the database with proper locking."""
    with db_lock:
        conn = get_db_connection(db_path)
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO predictions_reasoning
                (drug_id, drug_name, disease_id, disease_name, llm_explanation, 
                web_explanation, web_links, combined_confidence, source, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    evidence.drug_id,
                    evidence.drug_name,
                    evidence.disease_id,
                    evidence.disease_name,
                    evidence.llm_explanation,
                    evidence.web_explanation,
                    json.dumps(evidence.web_links),
                    evidence.combined_confidence,
                    evidence.source,
                    evidence.timestamp
                )
            )
            conn.commit()
        except Exception as e:
            logger.error(f"Database error saving evidence: {str(e)}")
        finally:
            conn.close()

def process_batch(pairs: List[DrugDiseasePair], ollama_url: str, serper_api_key: str, db_path: str) -> None:
    """Process a batch of pairs using a thread pool."""
    if not pairs:
        return
        
    processed_count = 0
    
    with ThreadPoolExecutor(max_workers=min(args.workers, len(pairs))) as executor:
        # Submit all pairs for processing
        futures = {executor.submit(combined_analysis, pair, ollama_url, serper_api_key): pair for pair in pairs}
        
        # Process results as they complete
        for future in as_completed(futures):
            pair = futures[future]
            try:
                evidence = future.result()
                
                # Save each result individually with lock protection
                save_evidence(db_path, evidence)
                
                processed_count += 1
                if processed_count % 5 == 0 or processed_count == len(pairs):
                    logger.info(f"Progress: {processed_count}/{len(pairs)} pairs processed in current batch")
                
            except Exception as e:
                logger.error(f"Failed to process {pair.drug_name} for {pair.disease_name}: {str(e)}")

def main():
    start_time = time.time()
    
    # Load config
    config_path = args.config_path
    if not os.path.isabs(config_path):
        config_path_project_root = os.path.join(project_root, config_path)
        if os.path.exists(config_path_project_root):
            config_path = config_path_project_root
    
    try:
        config = load_config(config_path)
        logger.info(f"Loaded configuration from {config_path}")
        
        # Extract settings from config
        db_path = config.get('ml_model', {}).get('db_path', 'data/processed/pubtator.db')
        if not os.path.isabs(db_path):
            db_path = os.path.join(project_root, db_path)
        
        serper_api_key = config.get('serper', {}).get('api_key')
        if not serper_api_key and not args.serper_api_key:
            logger.error("Serper API key not found in config or arguments")
            sys.exit(1)
        
        # Args override config
        serper_api_key = args.serper_api_key or serper_api_key
        
    except Exception as e:
        logger.error(f"Failed to load configuration: {str(e)}")
        if args.serper_api_key:
            logger.warning("Continuing with default database path")
            serper_api_key = args.serper_api_key
            db_path = os.path.join(project_root, 'data/processed/pubtator.db')
        else:
            sys.exit(1)
    
    # Set up database
    setup_database(db_path)
    logger.info(f"Database setup complete at {db_path}")
    
    # Fetch unprocessed pairs
    pairs = fetch_unprocessed_pairs(db_path, args.max_pairs)
    if not pairs:
        logger.info("No unprocessed pairs found. Exiting.")
        return
        
    logger.info(f"Found {len(pairs)} unprocessed pairs")
    
    # Process in smaller batches to allow for better progress tracking
    batch_size = args.batch_size
    batch_count = (len(pairs) + batch_size - 1) // batch_size  # Ceiling division
    
    for i in range(0, len(pairs), batch_size):
        batch = pairs[i:i+batch_size]
        batch_num = i // batch_size + 1
        logger.info(f"Processing batch {batch_num}/{batch_count} ({len(batch)} pairs)")
        
        process_batch(batch, args.ollama_url, serper_api_key, db_path)
        
        # Log progress after each batch
        elapsed = time.time() - start_time
        pairs_processed = min(i + batch_size, len(pairs))
        pairs_per_second = pairs_processed / elapsed if elapsed > 0 else 0
        estimated_total = elapsed * (len(pairs) / pairs_processed) if pairs_processed > 0 else 0
        remaining = estimated_total - elapsed if pairs_processed > 0 else 0
        
        logger.info(f"Completed {pairs_processed}/{len(pairs)} pairs")
        logger.info(f"Processing rate: {pairs_per_second:.2f} pairs/second")
        logger.info(f"Estimated time remaining: {datetime.timedelta(seconds=int(remaining))}")
    
    # Completion summary
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    logger.info(f"Completed evidence extraction for {len(pairs)} drug-disease pairs")
    logger.info(f"Total execution time: {datetime.timedelta(seconds=int(elapsed_time))}")
    logger.info(f"Average processing time per pair: {elapsed_time/len(pairs):.2f} seconds")

if __name__ == "__main__":
    main()