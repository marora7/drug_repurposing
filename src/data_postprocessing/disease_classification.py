import json
import requests
import os
import sys
import logging
import sqlite3
import datetime
import argparse
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


# Define enums for disease classifications
class DiseaseType(IntEnum):
    NOT_A_DISEASE = 0  # Added new category for non-disease inputs
    PRIMARY_TREATABLE = 1
    SYMPTOMS_COMPLICATIONS = 2
    STRUCTURAL_CONDITIONS = 3
    INTERVENTIONS_PROCEDURES = 4
    OTHER_MEDICAL_ENTITIES = 5
    
    @classmethod
    def get_name(cls, value: int) -> str:
        names = {
            0: "Not A Medical Condition",  # Name for non-disease category
            1: "Primary Treatable Diseases",
            2: "Symptoms/Complications",
            3: "Structural Conditions",
            4: "Interventions/Procedures",
            5: "Other Medical Entities"
        }
        return names.get(value, "Unknown")


class TherapeuticArea(IntEnum):
    ONCOLOGY = 1
    CARDIOVASCULAR = 2
    NEUROLOGY_CNS = 3
    INFECTIOUS_DISEASES = 4
    METABOLIC_ENDOCRINE = 5
    IMMUNOLOGY_INFLAMMATION = 6
    RESPIRATORY = 7
    GASTROENTEROLOGY = 8
    DERMATOLOGY = 9
    NEPHROLOGY_UROLOGY = 10
    HEMATOLOGY = 11
    MUSCULOSKELETAL = 12
    OPHTHALMOLOGY = 13
    PSYCHIATRY_MENTAL_HEALTH = 14
    RARE_DISEASES = 15
    UNCLASSIFIED = 16
    NOT_APPLICABLE = 17  # Added for non-disease inputs
    
    @classmethod
    def get_name(cls, value: int) -> str:
        names = {
            1: "Oncology",
            2: "Cardiovascular",
            3: "Neurology/CNS",
            4: "Infectious Diseases",
            5: "Metabolic/Endocrine",
            6: "Immunology/Inflammation",
            7: "Respiratory",
            8: "Gastroenterology",
            9: "Dermatology",
            10: "Nephrology/Urology",
            11: "Hematology",
            12: "Musculoskeletal",
            13: "Ophthalmology",
            14: "Psychiatry/Mental Health",
            15: "Rare Diseases",
            16: "Unclassified/Other",
            17: "Not Applicable"  # Name for the new category
        }
        return names.get(value, "Unknown")


# Define Pydantic model for disease classification response
class DiseaseClassification(BaseModel):
    disease_type: int = Field(..., ge=0, le=5)  
    disease_type_name: str
    therapeutic_area: int = Field(..., ge=1, le=17)  
    therapeutic_area_name: str
    reasoning: str
    confidence: float = Field(..., ge=0, le=1)
    
    @field_validator('disease_type_name')
    @classmethod
    def validate_disease_type_name(cls, v, info):
        values = info.data
        if 'disease_type' in values:
            expected_name = DiseaseType.get_name(values['disease_type'])
            if v != expected_name:
                raise ValueError(f"Expected disease_type_name to be {expected_name}")
        return v
    
    @field_validator('therapeutic_area_name')
    @classmethod
    def validate_therapeutic_area_name(cls, v, info):
        values = info.data
        if 'therapeutic_area' in values:
            expected_name = TherapeuticArea.get_name(values['therapeutic_area'])
            if v != expected_name:
                raise ValueError(f"Expected therapeutic_area_name to be {expected_name}")
        return v
    
    @field_validator('therapeutic_area')
    @classmethod
    def validate_therapeutic_area(cls, v, info):
        values = info.data
        # If it's NOT_A_DISEASE, therapeutic area should be NOT_APPLICABLE
        if 'disease_type' in values and values['disease_type'] == DiseaseType.NOT_A_DISEASE and v != TherapeuticArea.NOT_APPLICABLE:
            raise ValueError("Non-disease inputs must have therapeutic_area set to 17 (Not Applicable)")
        # If it's not Primary Treatable, therapeutic area should be UNCLASSIFIED (unless NOT_A_DISEASE)
        elif 'disease_type' in values and values['disease_type'] != DiseaseType.PRIMARY_TREATABLE and values['disease_type'] != DiseaseType.NOT_A_DISEASE and v != TherapeuticArea.UNCLASSIFIED:
            raise ValueError("Non-Primary Treatable Diseases must have therapeutic_area set to 16 (Unclassified/Other)")
        return v


# Define the prompt template
DISEASES_PROMPT = """
You are a medical classification expert. Classify the following disease/medical condition:
Disease ID: {disease_id}
Disease/condition name: {disease_name}

FIRST STEP - Determine if this is a medical condition at all:
If the input is NOT a medical condition or disease (e.g., foods, animals, general objects, activities), 
classify it as category 0 - "Not A Medical Condition".

If it IS a medical condition, then classify according to the types below:
DISEASE TYPE CLASSIFICATION:
0. Not A Medical Condition - Input is not a disease or medical condition at all (e.g., "ostrich meat", "football", "sunrise")
1. Primary Treatable Diseases - Medical conditions (including congenital ones) that are direct targets for drug therapy (e.g., autoimmune diabetes, neonatal diabetes)
2. Symptoms/Complications - Clinical manifestations or secondary effects of diseases (e.g., loss of awareness, diabetic coma)
3. Structural Conditions - Physical or anatomical abnormalities primarily addressed through surgical intervention rather than medication (e.g., cleft palate, certain heart defects)
4. Interventions/Procedures - Medical interventions or procedures (e.g., amputation, transplantation)
5. Other Medical Entities - Medical entities that don't fit clearly into the above categories

IMPORTANT: After determining the above classification, proceed based on the type:
- If category 0 (Not A Medical Condition): set therapeutic_area to 17 (Not Applicable)
- If category 1 (Primary Treatable Disease): classify into one of the therapeutic areas below (1-15)
- For all other categories (2-5): set therapeutic_area to 16 (Unclassified/Other)

THERAPEUTIC AREA CLASSIFICATION (For Primary Treatable Diseases ONLY):
1. Oncology - All types of cancers and related conditions
2. Cardiovascular - Heart diseases, stroke, hypertension, arrhythmias
3. Neurology/CNS - Neurological disorders, Alzheimer's, Parkinson's, epilepsy, multiple sclerosis
4. Infectious Diseases - Bacterial, viral, fungal infections, including HIV, tuberculosis, hepatitis
5. Metabolic/Endocrine - Diabetes, thyroid disorders, metabolic syndrome, obesity
6. Immunology/Inflammation - Autoimmune diseases, allergies, rheumatoid arthritis, psoriasis
7. Respiratory - Asthma, COPD, pulmonary fibrosis, pneumonia
8. Gastroenterology - IBD, IBS, GERD, liver diseases, pancreatic disorders
9. Dermatology - Skin conditions, eczema, acne, dermatitis
10. Nephrology/Urology - Kidney diseases, UTIs, renal failure, bladder disorders
11. Hematology - Blood disorders, anemia, leukemia, clotting disorders
12. Musculoskeletal - Bone and joint diseases, osteoporosis, arthritis
13. Ophthalmology - Eye diseases, glaucoma, macular degeneration
14. Psychiatry/Mental Health - Depression, anxiety, schizophrenia, bipolar disorder
15. Rare Diseases - Orphan diseases, genetic disorders
16. Unclassified/Other - For non-Primary Treatable Diseases
17. Not Applicable - For inputs that are not medical conditions at all

Respond in valid JSON format with the following structure:

If the input is NOT a medical condition:
{{
    "disease_type": 0,
    "disease_type_name": "Not A Medical Condition",
    "therapeutic_area": 17,
    "therapeutic_area_name": "Not Applicable",
    "reasoning": "This input is not a medical condition because...",
    "confidence": 0.9
}}

If the input is a Primary Treatable Disease:
{{
    "disease_type": 1,
    "disease_type_name": "Primary Treatable Diseases",
    "therapeutic_area": 6,  // Example: value 1-15 as appropriate
    "therapeutic_area_name": "Immunology/Inflammation",  // Corresponding name
    "reasoning": "Brief explanation for your classification",
    "confidence": 0.9
}}

If the input is a medical condition but NOT a Primary Treatable Disease:
{{
    "disease_type": 2,  // or 3, 4, 5 as appropriate
    "disease_type_name": "Symptoms/Complications",  // or the appropriate name
    "therapeutic_area": 16,
    "therapeutic_area_name": "Unclassified/Other",
    "reasoning": "Brief explanation for your classification",
    "confidence": 0.9
}}

IMPORTANT:
1. DO NOT return lists/arrays for any values - use single values only
2. For disease_type, use a single integer (0-5)
3. For therapeutic_area, ALWAYS use a single integer as specified above
4. For therapeutic_area_name, ALWAYS include the corresponding name for the therapeutic_area value
5. Make sure the entire JSON is valid
6. Do not include extra quotes within values
7. DO NOT use null values for any field
8. Be especially careful with items like foods, common objects, or activities - these are NOT medical conditions
"""


# Class to handle disease classification using Ollama
class DiseaseClassifier:
    def __init__(self, ollama_url: str = "http://localhost:11434/api/generate", timeout: int = 20):
        self.ollama_url = ollama_url
        self.model = "deepseek-r1"
        self.timeout = timeout  # Timeout in seconds
    
    def classify_disease(self, disease_id: str, disease_name: str) -> DiseaseClassification:
        """
        Classifies a disease using the Ollama LLM and returns a structured response.
        
        Args:
            disease_id (str): The ID of the disease
            disease_name (str): The name of the disease
            
        Returns:
            DiseaseClassification: A Pydantic model with the classification results
        """
        # Format the prompt
        prompt = DISEASES_PROMPT.format(disease_id=disease_id, disease_name=disease_name)
        
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
                    logger.error(f"JSON repair failed. Using default classification for {disease_name}")
                    classification_data = {
                        "disease_type": 1,  # Default to Primary Treatable
                        "disease_type_name": "Primary Treatable Diseases",
                        "therapeutic_area": 16,  # Unclassified 
                        "therapeutic_area_name": "Unclassified/Other",
                        "reasoning": f"Failed to parse model output for {disease_name}. Using default classification.",
                        "confidence": 0.5
                    }
            
            # Check for required fields and add defaults if missing
            required_fields = {
                "disease_type": 1,
                "disease_type_name": "Primary Treatable Diseases",
                "therapeutic_area": 16,
                "therapeutic_area_name": "Unclassified/Other",
                "reasoning": f"Classification for {disease_name}.",
                "confidence": 0.5
            }
            
            for field, default_value in required_fields.items():
                if field not in classification_data:
                    logger.warning(f"Required field '{field}' missing from LLM response. Using default value.")
                    classification_data[field] = default_value
            
            # Perform pre-validation fixes
            disease_type = classification_data.get("disease_type")
            
            # Handle non-disease inputs (type 0)
            if disease_type == DiseaseType.NOT_A_DISEASE:
                classification_data["therapeutic_area"] = TherapeuticArea.NOT_APPLICABLE
                classification_data["therapeutic_area_name"] = "Not Applicable"
            
            # Handle non-primary treatable disease inputs (type 2-5)
            elif disease_type != DiseaseType.PRIMARY_TREATABLE:
                classification_data["therapeutic_area"] = TherapeuticArea.UNCLASSIFIED
                classification_data["therapeutic_area_name"] = "Unclassified/Other"
                
            # Ensure therapeutic_area_name matches the therapeutic_area
            correct_area_name = TherapeuticArea.get_name(classification_data.get("therapeutic_area", 16))
            classification_data["therapeutic_area_name"] = correct_area_name
            
            # Ensure disease_type_name matches the disease_type
            correct_type_name = DiseaseType.get_name(classification_data.get("disease_type", 1))
            classification_data["disease_type_name"] = correct_type_name
            
            # Ensure confidence is a float between 0 and 1
            if not isinstance(classification_data.get("confidence"), float):
                try:
                    classification_data["confidence"] = float(classification_data.get("confidence", 0.5))
                except:
                    classification_data["confidence"] = 0.5
            
            # Clamp confidence to [0, 1]
            classification_data["confidence"] = max(0.0, min(1.0, classification_data["confidence"]))
            
            # Check for common non-disease items and force classification if needed
            common_non_diseases = [
                "meat", "food", "drink", "animal", "plant", "tree", "flower", "vehicle", 
                "car", "boat", "building", "furniture", "clothing", "sport", "game", 
                "movie", "book", "music", "art"
            ]
            
            if any(term in disease_name.lower() for term in common_non_diseases):
                # Double-check if it might be a non-disease
                if disease_type != DiseaseType.NOT_A_DISEASE:
                    # The model didn't classify it as a non-disease, but it probably is
                    if "confidence" in classification_data and classification_data["confidence"] < 0.8:
                        # Low confidence - override to NOT_A_DISEASE
                        classification_data["disease_type"] = DiseaseType.NOT_A_DISEASE
                        classification_data["disease_type_name"] = "Not A Medical Condition"
                        classification_data["therapeutic_area"] = TherapeuticArea.NOT_APPLICABLE
                        classification_data["therapeutic_area_name"] = "Not Applicable"
                        classification_data["reasoning"] = f"{disease_name} is likely not a medical condition but rather a common object/item."
            
            # Validate and create the Pydantic model
            try:
                return DiseaseClassification(**classification_data)
            except Exception as e:
                # If validation fails, log the error and try to fix the model
                logger.error(f"Validation error for {disease_name}: {e}")
                
                # Create a valid model with default values
                return DiseaseClassification(
                    disease_type=1,
                    disease_type_name="Primary Treatable Diseases",
                    therapeutic_area=16, 
                    therapeutic_area_name="Unclassified/Other",
                    reasoning=f"Failed to validate model for {disease_name}. Using default classification.",
                    confidence=0.5
                )
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error for {disease_name}: {e}")
            raise ConnectionError(f"Failed to connect to Ollama: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error for {disease_name}: {e}")
            raise Exception(f"Error during disease classification: {str(e)}")

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
    
    def fetch_diseases(self, input_table: str, query_template: str) -> List[Tuple[str, str]]:
        """
        Fetch diseases from the database.
        
        Args:
            input_table: The name of the input table
            query_template: The query template from config
            
        Returns:
            List of tuples containing disease_id and disease_name
        """
        query = query_template.format(table=input_table)
        
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query)
                return cursor.fetchall()
        except sqlite3.Error as e:
            logger.error(f"Database error while fetching diseases: {e}")
            raise
    
    def create_result_table(self):
        """Create the result table 'diseases_grouped' if it doesn't exist."""
        create_table_query = """
        CREATE TABLE IF NOT EXISTS diseases_grouped (
            disease_id TEXT PRIMARY KEY,
            disease_name TEXT,
            disease_type INTEGER,
            disease_type_name TEXT,
            therapeutic_area INTEGER,
            therapeutic_area_name TEXT,
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
    
    def save_classification(self, disease_id: str, disease_name: str, classification: DiseaseClassification):
        """
        Save the classification result to the database.
        
        Args:
            disease_id: The ID of the disease
            disease_name: The name of the disease
            classification: The classification result
        """
        insert_query = """
        INSERT OR REPLACE INTO diseases_grouped 
            (disease_id, disease_name, disease_type, disease_type_name, 
             therapeutic_area, therapeutic_area_name, confidence, reasoning)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(insert_query, (
                    disease_id,
                    disease_name,
                    classification.disease_type,
                    classification.disease_type_name,
                    classification.therapeutic_area,
                    classification.therapeutic_area_name,
                    classification.confidence,
                    classification.reasoning
                ))
                conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Database error while saving classification for {disease_name}: {e}")
            raise


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Disease Classification Pipeline')
    parser.add_argument('--config-path', type=str, default='src/config/config.yaml',
                        help='Path to the configuration file')
    return parser.parse_args()


def main():
    """Main function with improved error handling and batch processing."""
    # Start timing the execution
    start_time = datetime.datetime.now()
    logger.info(f"Started execution at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Disease Classification Pipeline')
    parser.add_argument('--config-path', type=str, default='src/config/config.yaml',
                        help='Path to the configuration file')
    parser.add_argument('--batch-size', type=int, default=50,
                        help='Number of diseases to process in each batch')
    parser.add_argument('--resume', action='store_true',
                        help='Resume processing from where it left off')
    parser.add_argument('--timeout', type=int, default=30,
                        help='Timeout in seconds for processing each disease')
    parser.add_argument('--max-retries', type=int, default=2,
                        help='Maximum number of retries for failed classifications')
    args = parser.parse_args()
    
    # Load config
    config_path = args.config_path
    try:
        config = load_config(config_path)
        logger.info(f"Loaded configuration from {config_path}")
        
        # Extract database settings from config
        db_path = config.get('ml_model', {}).get('db_path', 'data/processed/pubtator.db')
        
        # Extract disease classification settings
        input_table_name = config.get('disease_classification', {}).get('input_table', 'relations_after_prediction')
        get_diseases_query = config.get('classification_queries', {}).get('get_diseases_query', 
                                       "SELECT disease_id, disease_name FROM {table} GROUP BY disease_id")
        
        # Initialize the database handler
        db_handler = DatabaseHandler(db_path)
        
        # Create the result table
        db_handler.create_result_table()
        logger.info("Created or verified result table 'diseases_grouped'")
        
        # Get already processed diseases if resuming
        processed_disease_ids = set()
        if args.resume:
            try:
                with db_handler.get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT disease_id FROM diseases_grouped")
                    processed_disease_ids = {row[0] for row in cursor.fetchall()}
                logger.info(f"Found {len(processed_disease_ids)} already processed diseases")
            except sqlite3.Error as e:
                logger.error(f"Database error while fetching processed diseases: {e}")
        
        # Fetch diseases from database
        diseases = db_handler.fetch_diseases(input_table_name, get_diseases_query)
        total_diseases = len(diseases)
        logger.info(f"Fetched {total_diseases} diseases from table '{input_table_name}'")
        
        # Filter out already processed diseases if resuming
        if args.resume and processed_disease_ids:
            diseases = [(id, name) for id, name in diseases if id not in processed_disease_ids]
            logger.info(f"After filtering already processed diseases, {len(diseases)} remain to be processed")
        
        # Initialize the classifier
        classifier = DiseaseClassifier(timeout=args.timeout)
        logger.info(f"Initialized disease classifier with {args.timeout}s timeout")
        
        # Process in batches
        batch_size = args.batch_size
        total_batches = (len(diseases) + batch_size - 1) // batch_size  # Ceiling division
        
        # Track overall statistics
        overall_processed = 0
        overall_successful = 0
        overall_errors = 0
        overall_retries = 0
        
        # Create a file to track problematic diseases
        problematic_file = "problematic_diseases.txt"
        with open(problematic_file, "w") as f:
            f.write("disease_id,disease_name,error\n")
        
        # Process each batch
        for batch_idx in range(total_batches):
            batch_start = batch_idx * batch_size
            batch_end = min((batch_idx + 1) * batch_size, len(diseases))
            current_batch = diseases[batch_start:batch_end]
            
            logger.info(f"Processing batch {batch_idx+1}/{total_batches} ({batch_end-batch_start} diseases)")
            batch_start_time = datetime.datetime.now()
            
            # Batch statistics
            batch_processed = 0
            batch_successful = 0
            batch_errors = 0
            
            # Process each disease in the batch
            for disease_id, disease_name in current_batch:
                disease_start_time = datetime.datetime.now()
                logger.info(f"Processing disease: {disease_name} (ID: {disease_id})")
                
                # Set up retry logic
                retry_count = 0
                max_retries = args.max_retries
                success = False
                
                while retry_count <= max_retries and not success:
                    if retry_count > 0:
                        logger.info(f"Retry {retry_count}/{max_retries} for {disease_name}")
                        overall_retries += 1
                    
                    try:
                        # Use a timeout to prevent hanging
                        import signal
                        
                        def timeout_handler(signum, frame):
                            raise TimeoutError(f"Processing timed out after {args.timeout} seconds")
                        
                        # Set the timeout alarm
                        signal.signal(signal.SIGALRM, timeout_handler)
                        signal.alarm(args.timeout)
                        
                        try:
                            # Classify the disease
                            result = classifier.classify_disease(disease_id, disease_name)
                            
                            # Save the classification to the database
                            db_handler.save_classification(disease_id, disease_name, result)
                            
                            # Log the result
                            logger.info(f"Classified {disease_name} as {result.disease_type_name} (TA: {result.therapeutic_area_name})")
                            
                            batch_successful += 1
                            success = True
                            
                        except TimeoutError as te:
                            # Log the timeout and retry or continue
                            logger.error(f"Timeout processing disease {disease_name}: {te}")
                            with open(problematic_file, "a") as f:
                                f.write(f"{disease_id},{disease_name},\"Timeout: {te}\"\n")
                            
                        finally:
                            # Cancel the alarm
                            signal.alarm(0)
                        
                    except Exception as e:
                        logger.error(f"Error classifying disease {disease_name}: {e}")
                        with open(problematic_file, "a") as f:
                            f.write(f"{disease_id},{disease_name},\"{str(e)}\"\n")
                    
                    # Only retry if not successful
                    if not success:
                        retry_count += 1
                
                # Count as error if all retries failed
                if not success:
                    batch_errors += 1
                
                # Log processing time for this disease
                disease_end_time = datetime.datetime.now()
                disease_time = disease_end_time - disease_start_time
                logger.info(f"Processed {disease_name} in {format_time_delta(disease_time)}")
                
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
            progress_pct = (overall_processed / len(diseases)) * 100
            logger.info(f"Overall progress: {overall_processed}/{len(diseases)} ({progress_pct:.1f}%)")
            
            # Estimate time remaining
            elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
            if overall_processed > 0:
                avg_time_per_disease = elapsed_time / overall_processed
                remaining_diseases = len(diseases) - overall_processed
                est_time_remaining = remaining_diseases * avg_time_per_disease
                est_completion_time = datetime.datetime.now() + datetime.timedelta(seconds=est_time_remaining)
                
                logger.info(f"Estimated time remaining: {format_time_delta(datetime.timedelta(seconds=est_time_remaining))}")
                logger.info(f"Estimated completion time: {est_completion_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Log overall statistics
        logger.info(f"Processing complete!")
        logger.info(f"Total diseases processed: {overall_processed}")
        logger.info(f"Successful classifications: {overall_successful}")
        logger.info(f"Errors: {overall_errors}")
        logger.info(f"Total retries: {overall_retries}")
        
        # If any diseases were already processed (resumed run)
        if args.resume and processed_disease_ids:
            logger.info(f"Previously processed diseases: {len(processed_disease_ids)}")
            logger.info(f"Total diseases in database: {len(processed_disease_ids) + overall_successful}")
        
        # Log total execution time
        end_time = datetime.datetime.now()
        total_time = end_time - start_time
        logger.info(f"Total execution time: {format_time_delta(total_time)}")
        logger.info(f"Average time per disease: {format_time_delta(datetime.timedelta(seconds=total_time.total_seconds() / max(1, overall_processed)))}")
        logger.info(f"Script completed at {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Problematic diseases logged to {problematic_file}")
        
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        
        # Log execution time even if there was an error
        end_time = datetime.datetime.now()
        total_time = end_time - start_time
        logger.info(f"Execution terminated with error after {format_time_delta(total_time)}")

if __name__ == "__main__":
    main()
