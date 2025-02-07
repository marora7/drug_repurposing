"""
data_extraction package

This package contains modules for extracting data from various sources.
It provides the following public functions:
  - extract_pubtator: Extracts data from Pubtator.
  - extract_ncbi: Extracts human gene information from NCBI.
"""

from .extract_data_pubtator import main as extract_pubtator
from .extract_data_ncbi import main as extract_ncbi

__all__ = ['extract_pubtator', 'extract_ncbi']
