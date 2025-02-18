import argparse
import logging
from knowledge_graph.kg_importer import import_knowledge_graph

def main():
    parser = argparse.ArgumentParser(
        description="Offline Knowledge Graph Import Script for Neo4j"
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML configuration file (e.g., config/config.yaml)"
    )
    args = parser.parse_args()

    # Configure logging (you can adjust the path or level as needed)
    logging.basicConfig(
        filename='import_neo4j.log',
        filemode='w',  # Change to 'a' to append if desired.
        level=logging.DEBUG,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )

    import_knowledge_graph(args.config)

if __name__ == "__main__":
    main()
