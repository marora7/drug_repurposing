import os
import csv
import logging
import numpy as np
from typing import Dict, Union, List, Tuple

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