import json
import requests
import os
import sys
import logging
import sqlite3
import datetime
import argparse
import signal
from enum import IntEnum
from typing import Optional, Dict, Any, List, Tuple
from pydantic import BaseModel, Field, field_validator

# Get the absolute path of the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the project root directory (assuming the script is in src/data_postprocessing/)
project_root = os.path.abspath(os.path.join(current_dir, '../..'))
# Add the project root to sys.path
sys.path.insert(0, project_root)

from src.utils.config_utils import load_config

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Define enum for drug categories
class DrugCategory(IntEnum):
    APPROVED_DRUGS = 1
    NON_APPROVED_COMPOUNDS = 2
    VITAMINS_NUTRIENTS_SUPPLEMENTS = 3
    RESEARCH_TOOLS_REAGENTS = 4
    UNCLASSIFIED_OTHER = 5
    
    @classmethod
    def get_name(cls, value: int) -> str:
        names = {
            1: "Approved Drugs",
            2: "Non-Approved Compounds",
            3: "Vitamins/Nutrients/Supplements",
            4: "Research Tools/Reagents",
            5: "Unclassified/Other"
        }
        return names.get(value, "Unknown")


# Define Pydantic model for drug classification
class DrugClassification(BaseModel):
    drug_id: str
    drug_name: str
    category: int = Field(..., ge=1, le=5)
    category_name: str
    confidence: float = Field(..., ge=0, le=1)
    reasoning: str
    
    @field_validator('category_name')
    @classmethod
    def validate_category_name(cls, v, info):
        values = info.data
        if 'category' in values:
            expected_name = DrugCategory.get_name(values['category'])
            if v != expected_name:
                raise ValueError(f"Expected category_name to be {expected_name}")
        return v


# Define the prompt template
DRUGS_PROMPT = """
You are a pharmaceutical classification expert. Your task is to classify the following drug/compound into EXACTLY ONE of these five categories:

Drug ID: {drug_id}
Drug name: {drug_name}

STEP 1: Consider this classification system carefully:
1. Approved Drugs - Medications that have received formal approval from ANY regulatory agency worldwide (FDA, EMA, PMDA, MHRA, TGA, Health Canada, etc.) for human therapeutic use
2. Non-Approved Compounds - Experimental drugs in research or clinical trial phases, not yet approved for standard medical use in any major market
3. Vitamins/Nutrients/Supplements - Essential compounds, dietary supplements, minerals, amino acids, and natural products
4. Research Tools/Reagents - Compounds primarily used in laboratory settings for research purposes, not intended for human use
5. Unclassified/Other - Compounds that don't fit clearly into other categories or where insufficient information is available

STEP 2: Apply these specific guidelines:
- Approved Drugs: Include medications with formal approval from ANY recognized regulatory agency worldwide for human use
- Consider the primary intended use of the compound when making your classification
- For compounds with multiple uses, choose the most clinically relevant category
- If the compound is primarily used in research, classify as Research Tools/Reagents
- For natural products, consider whether they're used mainly as supplements or as approved medications

STEP 3: Analyze the drug and choose EXACTLY ONE category number (1-5)
- Think about what is known about this drug/compound
- Determine which single category best applies
- Do not select multiple categories or create new categories

STEP 4: Create a valid JSON response with this EXACT structure:
{{
    "drug_id": "{drug_id}",
    "drug_name": "{drug_name}",
    "category": 2,
    "category_name": "Non-Approved Compounds",
    "reasoning": "This compound is currently in Phase II clinical trials for cancer treatment but has not received regulatory approval in any major market.",
    "confidence": 0.85
}}

CRITICAL REQUIREMENTS:
1. category MUST be a SINGLE INTEGER between 1-5 (not a string, not an array)
2. category_name MUST be the EXACT corresponding name from the list above
3. confidence MUST be a SINGLE DECIMAL NUMBER between 0.0 and 1.0
4. DO NOT use lists or arrays for ANY field
5. DO NOT include comments (// etc.) in the final JSON
6. ENSURE JSON is valid and properly formatted
7. DO NOT create categories outside the five listed above
8. DO NOT include extra text before or after the JSON
9. ALWAYS include the drug_name in your response exactly as provided

STEP 5: Double-check your output!
- Verify category is a single integer (1, 2, 3, 4, or 5)
- Verify category_name matches exactly one of the five names listed
- Ensure there are NO arrays or lists in your response
- Ensure the JSON is valid with all required fields
- Make sure drug_name is included in the output
"""


# Class to handle drug classification using Ollama
class DrugClassifier:
    def __init__(self, ollama_url: str = "http://localhost:11434/api/generate", timeout: int = 20):
        self.ollama_url = ollama_url
        self.model = "deepseek-r1"
        self.timeout = timeout  # Timeout in seconds
    
    def classify_drug(self, drug_id: str, drug_name: str) -> DrugClassification:
        """
        Classifies a drug using the Ollama LLM and returns a structured response.
        
        Args:
            drug_id (str): The ID of the drug
            drug_name (str): The name of the drug
            
        Returns:
            DrugClassification: A Pydantic model with the classification results
        """
        # Format the prompt
        prompt = DRUGS_PROMPT.format(drug_id=drug_id, drug_name=drug_name)
        
        # Add a stronger JSON instruction to the prompt
        prompt += "\n\nIMPORTANT: Your response must be a single, well-formed JSON object. Do not include any text before or after the JSON."
        
        # Prepare the request to Ollama
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1  # Low temperature for more deterministic results
            }
        }
        
        try:
            # Send the request to Ollama with timeout
            response = requests.post(self.ollama_url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            
            # Extract the generated text
            result = response.json()
            generated_text = result.get("response", "")
            
            # Log a small preview of the response for debugging
            logger.debug(f"LLM response preview: {generated_text[:100]}...")
            
            # Extract the JSON part from the generated text
            json_start = generated_text.find('{')
            json_end = generated_text.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                raise ValueError("No JSON found in the response")
                
            json_text = generated_text[json_start:json_end]
            
            # Try to parse the JSON
            try:
                classification_data = json.loads(json_text)
            except json.JSONDecodeError as e:
                # If parsing fails, try to repair the JSON
                logger.warning(f"Initial JSON parsing failed: {e}. Attempting to repair JSON.")
                # Try to find valid JSON by looking for closest valid closing bracket
                # This is a simple heuristic; more sophisticated JSON repair could be implemented
                for i in range(len(json_text), json_start, -1):
                    try:
                        repaired_json = json_text[:i]
                        if repaired_json.count('{') == repaired_json.count('}'):
                            classification_data = json.loads(repaired_json)
                            logger.info("Successfully repaired JSON")
                            break
                    except:
                        continue
                else:
                    # If repair failed, create a default classification
                    logger.error(f"JSON repair failed. Using default classification for {drug_name}")
                    classification_data = {
                        "drug_id": drug_id,
                        "drug_name": drug_name,
                        "category": 5,  # Default to Unclassified
                        "category_name": "Unclassified/Other",
                        "reasoning": f"Failed to parse model output for {drug_name}. Using default classification.",
                        "confidence": 0.5
                    }
            
            # Check for required fields and add defaults if missing
            required_fields = {
                "drug_id": drug_id,
                "drug_name": drug_name,
                "category": 5,
                "category_name": "Unclassified/Other",
                "reasoning": f"Classification for {drug_name}.",
                "confidence": 0.5
            }
            
            for field, default_value in required_fields.items():
                if field not in classification_data:
                    logger.warning(f"Required field '{field}' missing from LLM response. Using default value.")
                    classification_data[field] = default_value
            
            # Ensure category is an integer between 1-5
            try:
                category = int(classification_data.get("category", 5))
                if category < 1 or category > 5:
                    logger.warning(f"Invalid category value {category}. Using default category 5.")
                    category = 5
                classification_data["category"] = category
            except:
                logger.warning(f"Non-integer category value. Using default category 5.")
                classification_data["category"] = 5
                
            # Ensure category_name matches the category
            correct_category_name = DrugCategory.get_name(classification_data.get("category", 5))
            classification_data["category_name"] = correct_category_name
            
            # Ensure confidence is a float between 0 and 1
            if not isinstance(classification_data.get("confidence"), float):
                try:
                    classification_data["confidence"] = float(classification_data.get("confidence", 0.5))
                except:
                    classification_data["confidence"] = 0.5
            
            # Clamp confidence to [0, 1]
            classification_data["confidence"] = max(0.0, min(1.0, classification_data["confidence"]))
            
            # Validate and create the Pydantic model
            try:
                return DrugClassification(**classification_data)
            except Exception as e:
                # If validation fails, log the error and try to fix the model
                logger.error(f"Validation error for {drug_name}: {e}")
                
                # Create a valid model with default values
                return DrugClassification(
                    drug_id=drug_id,
                    drug_name=drug_name,
                    category=5,
                    category_name="Unclassified/Other",
                    reasoning=f"Failed to validate model for {drug_name}. Using default classification.",
                    confidence=0.5
                )
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error for {drug_name}: {e}")
            raise ConnectionError(f"Failed to connect to Ollama: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error for {drug_name}: {e}")
            raise Exception(f"Error during drug classification: {str(e)}")


# Helper function to format time delta for logging
def format_time_delta(delta):
    total_seconds = delta.total_seconds()
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    return f"{int(hours)}h {int(minutes)}m {seconds:.2f}s"


# Database handler class
class DatabaseHandler:
    def __init__(self, db_path: str):
        self.db_path = db_path
        
    def get_connection(self):
        """Get a connection to the SQLite database."""
        return sqlite3.connect(self.db_path)
    
    def fetch_drugs(self, input_table: str, query_template: str) -> List[Tuple[str, str]]:
        """
        Fetch drugs from the database.
        
        Args:
            input_table: The name of the input table
            query_template: The query template from config
            
        Returns:
            List of tuples containing drug_id and drug_name
        """
        query = query_template.format(table=input_table)
        logger.info(f"Executing query: {query}")
        
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query)
                return cursor.fetchall()
        except sqlite3.Error as e:
            logger.error(f"Database error while fetching drugs: {e}")
            raise
    
    def create_result_table(self):
        """Create the result table 'drugs_grouped' if it doesn't exist."""
        create_table_query = """
        CREATE TABLE IF NOT EXISTS drugs_grouped (
            drug_id TEXT PRIMARY KEY,
            drug_name TEXT,
            category INTEGER,
            category_name TEXT,
            confidence REAL,
            reasoning TEXT
        )
        """
        
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(create_table_query)
                conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Database error while creating result table: {e}")
            raise
    
    def save_classification(self, classification: DrugClassification):
        """
        Save the classification result to the database.
        
        Args:
            classification: The classification result
        """
        insert_query = """
        INSERT OR REPLACE INTO drugs_grouped 
            (drug_id, drug_name, category, category_name, confidence, reasoning)
        VALUES (?, ?, ?, ?, ?, ?)
        """
        
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(insert_query, (
                    classification.drug_id,
                    classification.drug_name,
                    classification.category,
                    classification.category_name,
                    classification.confidence,
                    classification.reasoning
                ))
                conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Database error while saving classification for {classification.drug_name}: {e}")
            raise


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Drug Classification Pipeline')
    parser.add_argument('--config-path', type=str, default='src/config/config.yaml',
                        help='Path to the configuration file')
    parser.add_argument('--batch-size', type=int, default=50,
                        help='Number of drugs to process in each batch')
    parser.add_argument('--resume', action='store_true',
                        help='Resume processing from where it left off')
    parser.add_argument('--timeout', type=int, default=30,
                        help='Timeout in seconds for processing each drug')
    parser.add_argument('--max-retries', type=int, default=2,
                        help='Maximum number of retries for failed classifications')
    return parser.parse_args()


def main():
    """Main function with improved error handling and batch processing."""
    # Start timing the execution
    start_time = datetime.datetime.now()
    logger.info(f"Started execution at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Parse command line arguments
    args = parse_args()
    
    # Load config
    config_path = args.config_path
    try:
        config = load_config(config_path)
        logger.info(f"Loaded configuration from {config_path}")
        
        # Extract database settings from config
        db_path = config.get('ml_model', {}).get('db_path', 'data/processed/pubtator.db')
        
        # Extract drug classification settings
        input_table_name = config.get('drug_classification', {}).get('input_table', 'relations_after_prediction')
        get_drugs_query = config.get('classification_queries', {}).get('get_drugs_query', 
                                    "SELECT r.drug_id, r.drug_name FROM {table} r INNER JOIN diseases_grouped dg ON r.disease_id = dg.disease_id LEFT JOIN drugs_grouped drg ON r.drug_id = drg.drug_id WHERE dg.disease_type = 1 AND drg.drug_id is null GROUP BY r.drug_id")
        
        # Initialize the database handler
        db_handler = DatabaseHandler(db_path)
        
        # Create the result table
        db_handler.create_result_table()
        logger.info("Created or verified result table 'drugs_grouped'")
        
        # Get already processed drugs if resuming
        processed_drug_ids = set()
        if args.resume:
            try:
                with db_handler.get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT drug_id FROM drugs_grouped")
                    processed_drug_ids = {row[0] for row in cursor.fetchall()}
                logger.info(f"Found {len(processed_drug_ids)} already processed drugs")
            except sqlite3.Error as e:
                logger.error(f"Database error while fetching processed drugs: {e}")
        
        # Fetch drugs from database
        drugs = db_handler.fetch_drugs(input_table_name, get_drugs_query)
        total_drugs = len(drugs)
        logger.info(f"Fetched {total_drugs} drugs from table '{input_table_name}'")
        
        # Filter out already processed drugs if resuming
        if args.resume and processed_drug_ids:
            drugs = [(id, name) for id, name in drugs if id not in processed_drug_ids]
            logger.info(f"After filtering already processed drugs, {len(drugs)} remain to be processed")
        
        # Initialize the classifier
        classifier = DrugClassifier(timeout=args.timeout)
        logger.info(f"Initialized drug classifier with {args.timeout}s timeout")
        
        # Process in batches
        batch_size = args.batch_size
        total_batches = (len(drugs) + batch_size - 1) // batch_size  # Ceiling division
        
        # Track overall statistics
        overall_processed = 0
        overall_successful = 0
        overall_errors = 0
        overall_retries = 0
        
        # Create a file to track problematic drugs
        problematic_file = "problematic_drugs.txt"
        with open(problematic_file, "w") as f:
            f.write("drug_id,drug_name,error\n")
        
        # Process each batch
        for batch_idx in range(total_batches):
            batch_start = batch_idx * batch_size
            batch_end = min((batch_idx + 1) * batch_size, len(drugs))
            current_batch = drugs[batch_start:batch_end]
            
            logger.info(f"Processing batch {batch_idx+1}/{total_batches} ({batch_end-batch_start} drugs)")
            batch_start_time = datetime.datetime.now()
            
            # Batch statistics
            batch_processed = 0
            batch_successful = 0
            batch_errors = 0
            
            # Process each drug in the batch
            for drug_id, drug_name in current_batch:
                drug_start_time = datetime.datetime.now()
                logger.info(f"Processing drug: {drug_name} (ID: {drug_id})")
                
                # Set up retry logic
                retry_count = 0
                max_retries = args.max_retries
                success = False
                
                while retry_count <= max_retries and not success:
                    if retry_count > 0:
                        logger.info(f"Retry {retry_count}/{max_retries} for {drug_name}")
                        overall_retries += 1
                    
                    try:
                        # Set the timeout alarm
                        def timeout_handler(signum, frame):
                            raise TimeoutError(f"Processing timed out after {args.timeout} seconds")
                        
                        signal.signal(signal.SIGALRM, timeout_handler)
                        signal.alarm(args.timeout)
                        
                        try:
                            # Classify the drug
                            result = classifier.classify_drug(drug_id, drug_name)
                            
                            # Save the classification to the database
                            db_handler.save_classification(result)
                            
                            # Log the result
                            logger.info(f"Classified {drug_name} as {result.category_name} (confidence: {result.confidence})")
                            
                            batch_successful += 1
                            success = True
                            
                        except TimeoutError as te:
                            # Log the timeout and retry or continue
                            logger.error(f"Timeout processing drug {drug_name}: {te}")
                            with open(problematic_file, "a") as f:
                                f.write(f"{drug_id},{drug_name},\"Timeout: {te}\"\n")
                            
                        finally:
                            # Cancel the alarm
                            signal.alarm(0)
                        
                    except Exception as e:
                        logger.error(f"Error classifying drug {drug_name}: {e}")
                        with open(problematic_file, "a") as f:
                            f.write(f"{drug_id},{drug_name},\"{str(e)}\"\n")
                    
                    # Only retry if not successful
                    if not success:
                        retry_count += 1
                
                # Count as error if all retries failed
                if not success:
                    batch_errors += 1
                
                # Log processing time for this drug
                drug_end_time = datetime.datetime.now()
                drug_time = drug_end_time - drug_start_time
                logger.info(f"Processed {drug_name} in {format_time_delta(drug_time)}")
                
                batch_processed += 1
            
            # Update overall statistics
            overall_processed += batch_processed
            overall_successful += batch_successful
            overall_errors += batch_errors
            
            # Log batch statistics
            batch_end_time = datetime.datetime.now()
            batch_time = batch_end_time - batch_start_time
            logger.info(f"Batch {batch_idx+1}/{total_batches} completed in {format_time_delta(batch_time)}")
            logger.info(f"Batch stats - Processed: {batch_processed}, Successful: {batch_successful}, Errors: {batch_errors}")
            
            # Calculate and log overall progress
            progress_pct = (overall_processed / len(drugs)) * 100
            logger.info(f"Overall progress: {overall_processed}/{len(drugs)} ({progress_pct:.1f}%)")
            
            # Estimate time remaining
            elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
            if overall_processed > 0:
                avg_time_per_drug = elapsed_time / overall_processed
                remaining_drugs = len(drugs) - overall_processed
                est_time_remaining = remaining_drugs * avg_time_per_drug
                est_completion_time = datetime.datetime.now() + datetime.timedelta(seconds=est_time_remaining)
                
                logger.info(f"Estimated time remaining: {format_time_delta(datetime.timedelta(seconds=est_time_remaining))}")
                logger.info(f"Estimated completion time: {est_completion_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Log overall statistics
        logger.info(f"Processing complete!")
        logger.info(f"Total drugs processed: {overall_processed}")
        logger.info(f"Successful classifications: {overall_successful}")
        logger.info(f"Errors: {overall_errors}")
        logger.info(f"Total retries: {overall_retries}")
        
        # If any drugs were already processed (resumed run)
        if args.resume and processed_drug_ids:
            logger.info(f"Previously processed drugs: {len(processed_drug_ids)}")
            logger.info(f"Total drugs in database: {len(processed_drug_ids) + overall_successful}")
        
        # Log total execution time
        end_time = datetime.datetime.now()
        total_time = end_time - start_time
        logger.info(f"Total execution time: {format_time_delta(total_time)}")
        logger.info(f"Average time per drug: {format_time_delta(datetime.timedelta(seconds=total_time.total_seconds() / max(1, overall_processed)))}")
        logger.info(f"Script completed at {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Problematic drugs logged to {problematic_file}")
        
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        
        # Log execution time even if there was an error
        end_time = datetime.datetime.now()
        total_time = end_time - start_time
        logger.info(f"Execution terminated with error after {format_time_delta(total_time)}")

if __name__ == "__main__":
    main()