import os
import yaml
import logging

# Create a module-level logger
logger = logging.getLogger(__name__)

def load_config(config_path):
    """
    Loads a YAML configuration file.

    Args:
        config_path (str): The path to the YAML configuration file.
    
    Returns:
        dict: A dictionary with configuration parameters loaded from the YAML file.
    
    Raises:
        FileNotFoundError: If the configuration file does not exist.
    """
    if not os.path.exists(config_path):
        logger.error(f"Configuration file not found: {config_path}")
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    logger.info(f"Configuration loaded successfully from {config_path}")
    return config
