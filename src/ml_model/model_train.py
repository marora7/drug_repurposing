"""
This script sets up the environment and executes the DGL-KE training process 
for the drug repurposing project. It configures paths and environment variables 
before invoking the training command.
"""

import os
import subprocess
import sys
import pathlib

def main():
    # Generate training data
    subprocess.run(["python", "-m", "src.ml_model.model_train"], check=True)

    # Set DGLBACKEND environment variable for DGL-KE
    #os.environ["DGLBACKEND"] = "pytorch"

    # Build the command for DGL-KE training
    command = [
        "dglke_train",
        "--dataset", "drug_repurposing",
        "--data_path", "./data/processed/train",
        "--data_files", "train.tsv", "test.tsv", "val.tsv",
        "--format", "raw_udd_hrt",
        "--model_name", "TransE_l2",
        "--batch_size", "20480",
        "--neg_sample_size", "256",
        "--hidden_dim", "400",
        "--gamma", "12.0",
        "--lr", "0.1",
        "--max_step", "100000",
        "--log_interval", "1000",
        "--batch_size_eval", "16",
        "-adv",
        "--regularization_coef", "1.00E-07",
        "--num_thread", "6",
        "--gpu", "0",
        "--neg_sample_size_eval", "10000",
        "--has_edge_importance"
    ]

    # Run the training command
    subprocess.run(command, check=True)

if __name__ == '__main__':
    main()
