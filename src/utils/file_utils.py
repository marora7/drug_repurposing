import os
import logging

# Create a module-level logger
logger = logging.getLogger(__name__)

def ensure_directory_exists(directory):
    """
    Ensures that the specified directory exists. If it does not exist,
    the directory (and any necessary parent directories) will be created.
    
    Parameters:
        directory (str): The path to the directory to check/create.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"Directory created: {directory}")
    else:
        logger.info(f"Directory already exists: {directory}")

def log_progress(message, level=logging.INFO):
    """
    Logs a message using the module-level logger.
    
    Parameters:
        message (str): The message to be logged.
        level (int): The logging level (default is logging.INFO).
    """
    logger.log(level, message)

