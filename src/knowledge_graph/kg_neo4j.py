# Knowledge Graph Construction Script: Imports data into Neo4j using neo4j-admin.

import os
import subprocess
import logging
import gzip
import shutil
import argparse

from src.utils.config_utils import load_config

def decompress_gzip(source_gz, dest_csv):
    """
    Decompress a GZ file to CSV.
    """
    logger.info(f"Decompressing {source_gz} -> {dest_csv}")
    with gzip.open(source_gz, 'rb') as f_in, open(dest_csv, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
    logger.info(f"Decompression complete: {dest_csv}")

def run_neo4j_admin_import(neo4j_admin_path, database_name, nodes_csv, edges_csv):
    """
    Runs the neo4j-admin command to perform a full offline import into the specified database,
    overwriting any existing database with that name.
    """
    cmd = [
        neo4j_admin_path,
        "database",
        "import",
        "full",
        database_name,
        f"--nodes={nodes_csv}",
        f"--relationships={edges_csv}",
        "--overwrite-destination",
        "--verbose"
        # Additional flags (if needed) can be added here.
    ]

    logger.info("Constructed import command:")
    logger.info(" ".join(cmd))

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        logger.debug(f"neo4j-admin stdout:\n{result.stdout}")
        logger.debug(f"neo4j-admin stderr:\n{result.stderr}")

        if result.returncode == 0:
            logger.info("neo4j-admin import completed successfully.")
        else:
            logger.error(f"neo4j-admin import failed with return code {result.returncode}.")
            logger.error("Check the logs above (stderr) for details.")
    except Exception as e:
        logger.exception("An unexpected error occurred while running neo4j-admin import:")
        raise

def main():
    parser = argparse.ArgumentParser(
        description="Offline Knowledge Graph Import Script"
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML configuration file (e.g., config/config.yaml)"
    )
    args = parser.parse_args()

    # Load configuration and extract knowledge graph settings.
    config = load_config(args.config)
    kg_config = config.get("knowledge_graph", {})

    neo4j_bin_dir = kg_config.get("neo4j_bin_dir", 
        r"C:\Users\Manali Arora\.Neo4jDesktop\relate-data\dbmss\dbms-68825c1d-13da-4e5e-94df-454c0cafee7a\bin")
    neo4j_admin_bat = kg_config.get("neo4j_admin", "neo4j-admin.bat")
    neo4j_admin_path = os.path.join(neo4j_bin_dir, neo4j_admin_bat)

    nodes_csv_gz = kg_config.get("nodes_csv_gz", r"F:\datasets\nodes.csv.gz")
    edges_csv_gz = kg_config.get("edges_csv_gz", r"F:\datasets\edges.csv.gz")
    nodes_csv = kg_config.get("nodes_csv", r"F:\datasets\nodes.csv")
    edges_csv = kg_config.get("edges_csv", r"F:\datasets\edges.csv")
    database_name = kg_config.get("database_name", "drugrepurposing")

    logger.info("===== Starting Offline Import Script =====")
    print("Ensure Neo4j Desktop is stopped before proceeding.")
    input("Press Enter to continue or Ctrl+C to cancel...")

    # Decompress the CSV files from GZ if needed.
    decompress_gzip(nodes_csv_gz, nodes_csv)
    decompress_gzip(edges_csv_gz, edges_csv)

    logger.info("About to run neo4j-admin import (FULL, offline).")
    run_neo4j_admin_import(neo4j_admin_path, database_name, nodes_csv, edges_csv)

    logger.info("===== Offline Import Script Finished =====")
    print("Import finished. See 'import_neo4j.log' for details.")
    print("You can now restart Neo4j Desktop and view the imported database.")

if __name__ == "__main__":
    # Configure logging: all info and errors will be written to 'import_neo4j.log'
    logging.basicConfig(
        filename='import_neo4j.log',
        filemode='w',  # Change to 'a' to append instead of overwrite.
        level=logging.DEBUG,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )
    logger = logging.getLogger(__name__)
    main()
