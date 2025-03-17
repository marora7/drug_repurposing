#!/bin/bash

# Store the absolute project root path
PROJECT_ROOT=$(cd "$(dirname "$0")/../../" && pwd)

# Create logs directory using absolute path
mkdir -p "$PROJECT_ROOT/data/processed/logs"

# Change to project root
cd "$PROJECT_ROOT"

# Ensure the project root is in PYTHONPATH
export PYTHONPATH="$PROJECT_ROOT"

# Disable DGL GraphBolt (if not needed)
export DGL_GRAPHBOLT_PATH=""

# Run the training command and redirect output to log file
DGLBACKEND=pytorch dglke_train \
    --dataset drug_repurposing \
    --data_path "$PROJECT_ROOT/data/processed/train" \
    --data_files train.tsv val.tsv test.tsv \
    --save_path "$PROJECT_ROOT/data/processed/train" \
    --format 'raw_udd_hrt' \
    --model_name TransE_l2 \
    --batch_size 20480 \
    --neg_sample_size 256 \
    --hidden_dim 400 \
    --gamma 12.0 \
    --lr 0.1 \
    --max_step 100000 \
    --log_interval 1000 \
    --valid \
    --eval_interval 2000 \
    --batch_size_eval 4096 \
    --neg_sample_size_eval 10000 \
    --eval_percent 1.0 \
    --neg_adversarial_sampling \
    --regularization_coef 1.00E-07 \
    --num_thread 6 \
    --gpu 0 \
    --has_edge_importance 2>&1 | tee "$PROJECT_ROOT/data/processed/logs/training_$(date +%Y%m%d_%H%M%S).log"