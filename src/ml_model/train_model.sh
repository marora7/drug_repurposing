#!/bin/bash
# Change to project root (adjust the number of .. as needed)
cd ../..

# Ensure the project root is in PYTHONPATH
export PYTHONPATH=$(pwd)

# Disable DGL GraphBolt (if not needed)
export DGL_GRAPHBOLT_PATH=""

# First, generate the training data for your drug repurposing project
#python -m src.ml_model.ml_input_generation

# Now, train the model using DGL-KE.
# Make sure DGL-KE is installed and you have set up CUDA (if using GPU).
DGLBACKEND=pytorch dglke_train \
    --dataset drug_repurposing \
    --data_path ./data/processed/ml_input \
    --data_files train.tsv test.tsv val.tsv \
    --format 'raw_udd_hrt' \
    --model_name TransE_l2 \
    --batch_size 20480 \
    --neg_sample_size 256 \
    --hidden_dim 400 \
    --gamma 12.0 \
    --lr 0.1 \
    --max_step 100000 \
    --log_interval 1000 \
    --batch_size_eval 16 \
    -adv \
    --regularization_coef 1.00E-07 \
    --num_thread 6 \
    --gpu 0 \
    --neg_sample_size_eval 10000 \
    --has_edge_importance
