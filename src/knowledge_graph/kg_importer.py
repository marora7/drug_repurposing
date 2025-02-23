import os
import subprocess
import logging
import gzip
import shutil

from src.utils.config_utils import load_config

logger = logging.getLogger(__name__)

def decompress_gzip(source_gz: str, dest_csv: str) -> None:
    """
    Decompress a GZ file to CSV.
    """
    logger.info(f"Decompressing {source_gz} -> {dest_csv}")
    with gzip.open(source_gz, 'rb') as f_in, open(dest_csv, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
    logger.info(f"Decompression complete: {dest_csv}")

def run_neo4j_import(neo4j_admin_path: str, database_name: str, nodes_csv: str, edges_csv: str) -> None:
    """
    Run the neo4j-admin command to import data into the specified database.
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
    ]

    logger.info("Constructed import command: %s", " ".join(cmd))

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        logger.debug("neo4j-admin stdout:\n%s", result.stdout)
        logger.debug("neo4j-admin stderr:\n%s", result.stderr)

        if result.returncode == 0:
            logger.info("neo4j-admin import completed successfully.")
        else:
            logger.error("neo4j-admin import failed with return code %s.", result.returncode)
    except Exception as e:
        logger.exception("An error occurred while running neo4j-admin import:")
        raise

def import_knowledge_graph(config_path: str) -> None:
    """
    Main orchestration function for importing data into Neo4j.
    Loads configuration, prompts the user, decompresses the CSVs,
    and runs the import.
    """
    # Load configuration
    config = load_config(config_path)
    kg_config = config.get("knowledge_graph", {})

    # Construct paths from config
    neo4j_bin_dir = kg_config.get("neo4j_bin_dir")
    neo4j_admin_file = kg_config.get("neo4j_admin")
    neo4j_admin_path = os.path.join(neo4j_bin_dir, neo4j_admin_file)

    # Use exported files from data/exports (relative paths from git)
    nodes_csv_gz = kg_config.get("nodes_csv_gz")
    edges_csv_gz = kg_config.get("edges_csv_gz")
    nodes_csv = kg_config.get("nodes_csv")
    edges_csv = kg_config.get("edges_csv")
    database_name = kg_config.get("database_name")

    logger.info("===== Starting Neo4j Offline Import =====")
    print("Ensure Neo4j Desktop is stopped before proceeding.")
    input("Press Enter to continue or Ctrl+C to cancel...")

    # Decompress the CSV files if needed
    decompress_gzip(nodes_csv_gz, nodes_csv)
    decompress_gzip(edges_csv_gz, edges_csv)

    logger.info("Running neo4j-admin import.")
    run_neo4j_import(neo4j_admin_path, database_name, nodes_csv, edges_csv)
    logger.info("===== Neo4j Offline Import Finished =====")
