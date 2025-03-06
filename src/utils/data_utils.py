import os
import csv
import logging
import sqlite3
import requests
import json
import re
import numpy as np
from typing import Dict, Union, List, Tuple
from pydantic import BaseModel
from typing import Optional
from typing_extensions import Literal

logger = logging.getLogger(__name__)

def load_tsv_mappings(tsv_file: str, label: str = "items") -> Tuple[List[str], Dict[str, int]]:
    """
    Loads items from a TSV file and creates an ordered list and a lookup dictionary.
    
    Args:
        tsv_file (str): Path to the TSV file with two columns: index and item name
        label (str, optional): Label for logging purposes. Defaults to "items".
    
    Returns:
        Tuple[List[str], Dict[str, int]]: A tuple containing:
            - List of items in order
            - Dictionary mapping item names to their indices
    """
    items = []
    item_to_idx = {}
    
    logging.info(f"Loading {label} from {tsv_file}")
    with open(tsv_file, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            if len(row) < 2:
                continue
            idx, item = row[0], row[1]
            items.append(item)
            item_to_idx[item] = int(idx)
    
    logging.info(f"Loaded {len(items)} {label}")
    return items, item_to_idx

def load_embeddings(file_paths: Union[str, Dict[str, str], List[str]], 
                    names: List[str] = None) -> Union[np.ndarray, Tuple[np.ndarray, ...], Dict[str, np.ndarray]]:
    """
    Loads embedding matrices from .npy files.
    
    Args:
        file_paths: Either a single path string, a list of paths, or a dictionary mapping names to paths
        names: Optional list of names when file_paths is a list
    
    Returns:
        Either a single numpy array, a tuple of arrays, or a dictionary mapping names to arrays
    
    Examples:
        # Load a single embedding file
        entity_emb = load_embeddings('path/to/entity_embeddings.npy')
        
        # Load multiple embedding files with custom names
        embeddings = load_embeddings({
            'entity': 'path/to/entity_embeddings.npy',
            'relation': 'path/to/relation_embeddings.npy'
        })
        
        # Load multiple embedding files as a tuple
        entity_emb, relation_emb = load_embeddings([
            'path/to/entity_embeddings.npy',
            'path/to/relation_embeddings.npy'
        ])
    """
    # Case 1: Single file path
    if isinstance(file_paths, str):
        if not os.path.exists(file_paths):
            raise FileNotFoundError(f"Embedding file not found: {file_paths}")
        
        logging.info(f"Loading embeddings from {file_paths}")
        return np.load(file_paths)
    
    # Case 2: Dictionary of name -> path
    if isinstance(file_paths, dict):
        embeddings = {}
        for name, path in file_paths.items():
            if not os.path.exists(path):
                raise FileNotFoundError(f"{name} embedding file not found: {path}")
            
            logging.info(f"Loading {name} embeddings from {path}")
            embeddings[name] = np.load(path)
        
        return embeddings
    
    # Case 3: List of paths
    if isinstance(file_paths, (list, tuple)):
        result = []
        for i, path in enumerate(file_paths):
            if not os.path.exists(path):
                name = names[i] if names and i < len(names) else f"embeddings[{i}]"
                raise FileNotFoundError(f"{name} embedding file not found: {path}")
            
            name = names[i] if names and i < len(names) else f"embeddings[{i}]"
            logging.info(f"Loading {name} from {path}")
            result.append(np.load(path))
        
        return tuple(result)
    
    raise ValueError("file_paths must be a string, a dictionary, or a list")

# Define Pydantic models for structured output
class DrugClassification(BaseModel):
    drug_id: str
    drug_name: str
    category: Literal[1, 2, 3, 4, 5]
    category_name: str
    confidence: Optional[float] = None
    reasoning: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "drug_id": "Chemical:MESH:D010068",
                "drug_name": "oxacillin",
                "category": 1,
                "category_name": "FDA-Approved Drugs",
                "confidence": 0.9,
                "reasoning": "Well-established medication approved for human use"
            }
        }

def get_entities_from_db(db_path, config, entity_type):
    """
    Get entities (drugs, diseases, etc.) from the database based on entity_type.
    
    Args:
        db_path (str): Path to the SQLite database
        config (dict): Configuration dictionary from config.yaml
        entity_type (str): Type of entity to retrieve ('drug' or 'disease')
    
    Returns:
        list: List of tuples containing (entity_id, entity_name)
    """
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get query from config based on entity_type
        if entity_type == 'drug':
            query = config.get('classification_queries', {}).get('get_drugs_query')
            table_name = config.get('drug_classification', {}).get('input_table', 'relations_after_prediction')
        elif entity_type == 'disease':
            query = config.get('classification_queries', {}).get('get_diseases_query')
            table_name = config.get('disease_classification', {}).get('input_table', 'relations_after_prediction')
        else:
            logger.error(f"Unknown entity type: {entity_type}")
            return []
        
        # Replace placeholder with actual table name if needed
        if query and '{table}' in query:
            query = query.format(table=table_name)
        
        logger.debug(f"Executing query: {query}")
        cursor.execute(query)
        results = cursor.fetchall()
        logger.info(f"Retrieved {len(results)} {entity_type}s from database")
        return results
    
    except Exception as e:
        logger.error(f"Error retrieving {entity_type}s from database: {e}")
        return []
    
    finally:
        if conn:
            conn.close()

def extract_json_fields(json_string):
    """Extract fields from a JSON string safely."""
    try:
        # Try standard JSON parsing first
        data = json.loads(json_string)
        
        # Handle case where category or confidence might be returned as a list
        category = data.get('category', 5)
        if isinstance(category, list) and len(category) > 0:
            category = category[0]
        
        confidence = data.get('confidence', 0.5)
        if isinstance(confidence, list) and len(confidence) > 0:
            confidence = confidence[0]
            
        return {
            'drug_id': data.get('drug_id', ''),
            'category': int(category),
            'category_name': data.get('category_name', ''),
            'reasoning': data.get('reasoning', ''),
            'confidence': float(confidence)
        }
    except json.JSONDecodeError:
        # Fall back to regex for extracting fields
        fields = {}
        
        # Extract drug_id
        drug_id_match = re.search(r'"drug_id":\s*"([^"]+)"', json_string)
        fields['drug_id'] = drug_id_match.group(1).strip() if drug_id_match else ""
        
        # Extract category (number between 1-5)
        # Handle both direct values and array values
        category_array_match = re.search(r'"category":\s*\[(\d)\]', json_string)
        category_direct_match = re.search(r'"category":\s*(\d)', json_string)
        
        if category_array_match:
            fields['category'] = int(category_array_match.group(1))
        elif category_direct_match:
            fields['category'] = int(category_direct_match.group(1))
        else:
            fields['category'] = 5
        
        # Extract category_name
        category_name_match = re.search(r'"category_name":\s*"([^"]+)"', json_string)
        fields['category_name'] = category_name_match.group(1).strip() if category_name_match else ""
        
        # Extract reasoning - use a pattern that stops at the next field
        reasoning_pattern = r'"reasoning":\s*"([^"]*(?:\\"|[^"])*?)"(?:,|\s*})'
        reasoning_match = re.search(reasoning_pattern, json_string, re.DOTALL)
        fields['reasoning'] = reasoning_match.group(1).strip() if reasoning_match else ""
        
        # Extract confidence - handle both direct values and array values
        confidence_array_match = re.search(r'"confidence":\s*\[(0\.\d+|1\.0|1|0)\]', json_string)
        confidence_direct_match = re.search(r'"confidence":\s*(0\.\d+|1\.0|1|0)', json_string)
        
        if confidence_array_match:
            fields['confidence'] = float(confidence_array_match.group(1))
        elif confidence_direct_match:
            fields['confidence'] = float(confidence_direct_match.group(1))
        else:
            fields['confidence'] = 0.5
        
        return fields

def classify_drug(drug_id, drug_name, config):
    """Query Ollama to classify a single drug and return structured result"""
    
    # Get LLM configuration from config
    llm_config = config.get('llm_configuration', {})
    api_url = llm_config.get('api_url', 'http://localhost:11434/api/generate')
    model_name = llm_config.get('model_name', 'llama3.2')
    
    # Get the prompt template from config
    prompt_template = config.get('drugs_prompt', '')
    
    # Format the prompt with drug information
    prompt = prompt_template.format(drug_id=drug_id, drug_name=drug_name)
    
    # Define the categories for reference
    categories = {
        1: "FDA-Approved Drugs",
        2: "Non-Approved Compounds",
        3: "Vitamins/Nutrients/Supplements",
        4: "Research Tools/Reagents",
        5: "Unclassified/Other"
    }
    
    # Prepare the payload for Ollama API
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False
    }
    
    try:
        # Call the Ollama API
        logger.info(f"Sending request to classify: {drug_id} - {drug_name}")
        response = requests.post(api_url, json=payload)
        response.raise_for_status()
        result = response.json()
        
        # Extract the LLM's response
        llm_response = result['response'].strip()
        logger.debug(f"Raw LLM response:\n{llm_response}")
        
        # Find a JSON object in the response
        json_start = llm_response.find('{')
        json_end = llm_response.rfind('}') + 1
        
        if json_start >= 0 and json_end > json_start:
            json_str = llm_response[json_start:json_end]
            logger.debug(f"Extracted JSON string: {json_str}")
            
            # Extract fields from the JSON
            fields = extract_json_fields(json_str)
            logger.debug(f"Extracted fields: {fields}")
            
            # Create the classification
            classification = DrugClassification(
                drug_id=drug_id,  # Use passed drug_id as default
                drug_name=drug_name,
                category=fields['category'],
                category_name=fields['category_name'],
                confidence=fields['confidence'],
                reasoning=fields['reasoning']
            )
            
            logger.info(f"Final classification for {drug_name}: category={classification.category}, confidence={classification.confidence}")
            return classification
        else:
            # No JSON found in response
            logger.warning("No JSON object found in response")
            return DrugClassification(
                drug_id=drug_id,
                drug_name=drug_name,
                category=5,
                category_name="Unclassified/Other",
                confidence=0.0,
                reasoning=f"Could not extract classification data from model response"
            )
                
    except Exception as e:
        logger.error(f"Error classifying {drug_name}: {e}")
        # Return default classification on error
        return DrugClassification(
            drug_id=drug_id,
            drug_name=drug_name,
            category=5,
            category_name="Unclassified/Other",
            confidence=0.0,
            reasoning=f"Error: {str(e)}"
        )
    
# Define Pydantic models for structured output
class DiseaseClassification(BaseModel):
    disease_id: str
    disease_name: str
    # Disease Type Classification
    disease_type: Literal[1, 2, 3, 4, 5]
    disease_type_name: str
    # Therapeutic Area Classification - defaulting to Unclassified/Other for non-Primary Treatable Diseases
    therapeutic_area: Literal[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16] = 16
    therapeutic_area_name: str = "Unclassified/Other"
    confidence: Optional[float] = None
    reasoning: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "disease_id": "Disease:MESH:D003922",
                "disease_name": "Diabetes",
                "disease_type": 1,
                "disease_type_name": "Primary Treatable Diseases",
                "therapeutic_area": 5,
                "therapeutic_area_name": "Metabolic/Endocrine",
                "confidence": 0.9,
                "reasoning": "Chronic condition primarily treated with medication"
            }
        }

def extract_json_fields_disease(json_string):
    """Extract fields from a disease classification JSON string safely."""
    try:
        # Try standard JSON parsing first
        data = json.loads(json_string)
        
        # Handle disease_type - might be a list or a single value
        disease_type = data.get('disease_type', 5)
        if isinstance(disease_type, list) and len(disease_type) > 0:
            disease_type = disease_type[0]
        
        # Process therapeutic area fields - default to 16/"Unclassified/Other"
        therapeutic_area = 16
        therapeutic_area_name = "Unclassified/Other"
        
        # Extract therapeutic_area if present (for both Primary Treatable and non-Primary Treatable diseases)
        ta_value = data.get('therapeutic_area', 16)
        if ta_value is not None:
            if isinstance(ta_value, list) and len(ta_value) > 0:
                therapeutic_area = int(ta_value[0]) if ta_value[0] is not None else 16
            else:
                therapeutic_area = int(ta_value) if ta_value is not None else 16
        
        # Extract therapeutic_area_name if present
        ta_name = data.get('therapeutic_area_name', "Unclassified/Other")
        if ta_name is not None:
            if isinstance(ta_name, list) and len(ta_name) > 0:
                therapeutic_area_name = ta_name[0] if ta_name[0] is not None else "Unclassified/Other"
            else:
                therapeutic_area_name = ta_name if ta_name is not None else "Unclassified/Other"
            
            if isinstance(therapeutic_area_name, str) and therapeutic_area_name.startswith('"') and therapeutic_area_name.endswith('"'):
                therapeutic_area_name = therapeutic_area_name[1:-1]
        
        # Handle disease_type_name - might have extra quotes
        disease_type_name = data.get('disease_type_name', '')
        if isinstance(disease_type_name, str) and disease_type_name.startswith('"') and disease_type_name.endswith('"'):
            disease_type_name = disease_type_name[1:-1]
        
        # Handle confidence - might be a list or a single value
        confidence = data.get('confidence', 0.5)
        if isinstance(confidence, list) and len(confidence) > 0:
            confidence = confidence[0]
        
        # Handle reasoning - might have extra quotes
        reasoning = data.get('reasoning', '')
        if isinstance(reasoning, str) and reasoning.startswith('"') and reasoning.endswith('"'):
            reasoning = reasoning[1:-1]
        
        return {
            'disease_type': int(disease_type),
            'disease_type_name': disease_type_name,
            'therapeutic_area': int(therapeutic_area),
            'therapeutic_area_name': therapeutic_area_name,
            'reasoning': reasoning,
            'confidence': float(confidence)
        }
    except (json.JSONDecodeError, ValueError, TypeError) as e:
        logger.error(f"JSON parsing error: {e}")
        # Fall back to regex for extracting fields
        fields = {}
        
        # Extract disease_type (number between 1-5)
        disease_type_match = re.search(r'"disease_type":\s*(\d)', json_string)
        disease_type = int(disease_type_match.group(1)) if disease_type_match else 5
        fields['disease_type'] = disease_type
        
        # Extract disease_type_name
        disease_type_name_match = re.search(r'"disease_type_name":\s*"([^"]+)"', json_string)
        fields['disease_type_name'] = disease_type_name_match.group(1).strip() if disease_type_name_match else ""
        
        # Default to Unclassified/Other for therapeutic area
        fields['therapeutic_area'] = 16
        fields['therapeutic_area_name'] = "Unclassified/Other"
        
        # Try to extract therapeutic_area if present (for both types of diseases)
        therapeutic_area_match = re.search(r'"therapeutic_area":\s*(\d+)', json_string)
        if therapeutic_area_match:
            fields['therapeutic_area'] = int(therapeutic_area_match.group(1))
        
        # Try to extract therapeutic_area_name if present
        therapeutic_area_name_match = re.search(r'"therapeutic_area_name":\s*"([^"]+)"', json_string)
        if therapeutic_area_name_match:
            fields['therapeutic_area_name'] = therapeutic_area_name_match.group(1).strip()
        
        # Extract reasoning - use a pattern that stops at the next field
        reasoning_pattern = r'"reasoning":\s*"([^"]*(?:\\"|[^"])*?)"(?:,|\s*})'
        reasoning_match = re.search(reasoning_pattern, json_string, re.DOTALL)
        fields['reasoning'] = reasoning_match.group(1).strip() if reasoning_match else ""
        
        # Extract confidence
        confidence_pattern = r'"confidence":\s*(0\.\d+|1\.0|1|0)'
        confidence_match = re.search(confidence_pattern, json_string)
        fields['confidence'] = float(confidence_match.group(1)) if confidence_match else 0.5
        
        return fields

def classify_disease(disease_id, disease_name, config):
    """Query Ollama to classify a disease and return structured result"""
    
    # Get LLM configuration from config
    llm_config = config.get('llm_configuration', {})
    api_url = llm_config.get('api_url', 'http://localhost:11434/api/generate')
    model_name = llm_config.get('model_name', 'llama3.2')
    
    # Get the prompt template from config
    prompt_template = config.get('diseases_prompt', '')
    
    # Format the prompt with disease information
    prompt = prompt_template.format(disease_id=disease_id, disease_name=disease_name)
    
    # Define the disease types for reference
    disease_types = {
        1: "Primary Treatable Diseases",
        2: "Symptoms/Complications",
        3: "Structural Conditions",
        4: "Interventions/Procedures",
        5: "Other Medical Entities"
    }
    
    # Define the therapeutic areas for reference
    therapeutic_areas = {
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
        16: "Unclassified/Other"
    }
    
    # Prepare the payload for Ollama API
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False
    }
    
    try:
        # Call the Ollama API
        logger.info(f"Sending request to classify: {disease_id} - {disease_name}")
        response = requests.post(api_url, json=payload)
        response.raise_for_status()
        result = response.json()
        
        # Extract the LLM's response
        llm_response = result['response'].strip()
        logger.debug(f"Raw LLM response:\n{llm_response}")
        
        # Find a JSON object in the response
        json_start = llm_response.find('{')
        json_end = llm_response.rfind('}') + 1
        
        if json_start >= 0 and json_end > json_start:
            json_str = llm_response[json_start:json_end]
            logger.debug(f"Extracted JSON string: {json_str}")
            
            # Extract fields from the JSON
            fields = extract_json_fields_disease(json_str)
            logger.debug(f"Extracted fields: {fields}")
            
            # Create the classification with all fields
            classification = DiseaseClassification(
                disease_id=disease_id,
                disease_name=disease_name,
                disease_type=fields['disease_type'],
                disease_type_name=fields['disease_type_name'],
                therapeutic_area=fields['therapeutic_area'],
                therapeutic_area_name=fields['therapeutic_area_name'],
                confidence=fields['confidence'],
                reasoning=fields['reasoning']
            )
            
            logger.info(f"Final classification: disease_type={classification.disease_type}, therapeutic_area={classification.therapeutic_area}, confidence={classification.confidence}")
            return classification
        else:
            # No JSON found in response
            logger.warning("No JSON object found in response")
            return DiseaseClassification(
                disease_id=disease_id,
                disease_name=disease_name,
                disease_type=5,
                disease_type_name="Other Medical Entities",
                therapeutic_area=16,
                therapeutic_area_name="Unclassified/Other",
                confidence=0.0,
                reasoning=f"Could not extract classification data from model response"
            )
                
    except Exception as e:
        logger.error(f"Error classifying {disease_name}: {e}")
        # Return default classification on error
        return DiseaseClassification(
            disease_id=disease_id,
            disease_name=disease_name,
            disease_type=5,
            disease_type_name="Other Medical Entities",
            therapeutic_area=16,
            therapeutic_area_name="Unclassified/Other",
            confidence=0.0,
            reasoning=f"Error: {str(e)}"
        )