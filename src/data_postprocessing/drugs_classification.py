# Import necessary libraries
import time
import os
import sys
import logging
import argparse
import datetime

# Get the absolute path of the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the project root directory (assuming the script is in src/data_postprocessing/)
project_root = os.path.abspath(os.path.join(current_dir, '../..'))
# Add the project root to sys.path
sys.path.insert(0, project_root)

from src.utils.config_utils import load_config
from src.utils.data_utils import get_entities_from_db, classify_drug
from src.utils.db_utils import create_drugs_grouping, drugs_classification_to_db

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Classify drugs using LLM and store results in database')
    parser.add_argument('--config-path', type=str, default='src/config/config.yaml',
                        help='Path to the configuration file')
    return parser.parse_args()

def format_time_delta(delta):
    """Format timedelta to HH:MM:SS.ms format"""
    hours, remainder = divmod(delta.total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{seconds:06.3f}"

def main():
    """Main function to execute the script."""
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
        input_table = config.get('drug_classification', {}).get('input_table', 'relations_after_prediction')
                
        # Create drugs_grouping table if it doesn't exist
        table_creation_start = datetime.datetime.now()
        create_drugs_grouping(db_path, config)
        logger.info(f"Table creation completed in {format_time_delta(datetime.datetime.now() - table_creation_start)}")
        
        # Get the list of drugs from the database using the utility function
        fetch_start = datetime.datetime.now()
        drugs = get_entities_from_db(db_path, config, 'drug')
        logger.info(f"Found {len(drugs)} drugs to classify in {format_time_delta(datetime.datetime.now() - fetch_start)}")
        
        # Process each drug
        classification_start = datetime.datetime.now()
        processed_count = 0
        
        for drug_id, drug_name in drugs:
            drug_start_time = datetime.datetime.now()
            logger.info(f"Processing drug {processed_count + 1}/{len(drugs)}: {drug_id} - {drug_name}")
            
            # Classify the drug using the function from data_utils.py
            classification = classify_drug(drug_id, drug_name, config)
            
            # Save the classification to the database using the utility function
            drugs_classification_to_db(db_path, classification, config)
            
            processed_count += 1
            drug_processing_time = datetime.datetime.now() - drug_start_time
            logger.info(f"Processed drug in {format_time_delta(drug_processing_time)}")
            
            # Calculate and log estimated time remaining
            if processed_count > 0:
                avg_time_per_drug = (datetime.datetime.now() - classification_start).total_seconds() / processed_count
                remaining_drugs = len(drugs) - processed_count
                estimated_time_remaining = datetime.timedelta(seconds=avg_time_per_drug * remaining_drugs)
                logger.info(f"Estimated time remaining: {format_time_delta(estimated_time_remaining)}")
            
            # Small delay to prevent overloading the Ollama API
            time.sleep(1)
        
        classification_time = datetime.datetime.now() - classification_start
        logger.info(f"Classification of {processed_count} drugs completed in {format_time_delta(classification_time)}")
        
        # Log total execution time
        end_time = datetime.datetime.now()
        total_time = end_time - start_time
        logger.info(f"Total execution time: {format_time_delta(total_time)}")
        logger.info(f"Average time per drug: {format_time_delta(datetime.timedelta(seconds=total_time.total_seconds() / max(1, processed_count)))}")
        logger.info(f"Script completed at {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        
        # Log execution time even if there was an error
        end_time = datetime.datetime.now()
        total_time = end_time - start_time
        logger.info(f"Execution terminated with error after {format_time_delta(total_time)}")

if __name__ == "__main__":
    main()