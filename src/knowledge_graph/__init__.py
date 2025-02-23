"""
knowledge_graph package

This package contains modules for constructing and managing the Neo4j knowledge graph.
It provides the following public function:
  - import_graph: Executes the offline import process into Neo4j.
"""

from .kg_neo4j import main as import_graph

__all__ = ['import_graph']
