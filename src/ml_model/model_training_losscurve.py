"""
This script parses DGL-KE training logs and generates a plot of the training loss.
Run this after your training is complete.
"""

import os
import re
import matplotlib.pyplot as plt
import numpy as np

def extract_loss_from_logs(log_path):
    """
    Extract training loss values from DGL-KE log files.
    """
    loss_pattern = r"Training average loss: ([\d\.]+)"
    step_pattern = r"Step: (\d+)"
    
    steps = []
    losses = []
    
    # Find the training log file
    log_files = [f for f in os.listdir(log_path) if f.endswith('.log')]
    
    if not log_files:
        print(f"No log files found in {log_path}")
        return steps, losses
    
    # Use the most recent log file if there are multiple
    log_file = os.path.join(log_path, sorted(log_files)[-1])
    print(f"Parsing log file: {log_file}")
    
    with open(log_file, 'r') as f:
        log_content = f.read()
    
    # Find all loss entries with their steps
    step_matches = re.findall(step_pattern, log_content)
    loss_matches = re.findall(loss_pattern, log_content)
    
    # Make sure we have the same number of steps and losses
    min_len = min(len(step_matches), len(loss_matches))
    
    for i in range(min_len):
        steps.append(int(step_matches[i]))
        losses.append(float(loss_matches[i]))
    
    return steps, losses

def plot_training_loss(steps, losses, output_path):
    """
    Create and save a plot of the training loss.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(steps, losses, marker='o', linestyle='-', markersize=3)
    plt.title('Training Loss over Time')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add a smoothed trend line
    if len(steps) > 5:
        try:
            window_size = max(3, len(steps) // 20)  # Use a reasonable window size
            smoothed = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')
            valid_steps = steps[window_size-1:][:len(smoothed)]
            plt.plot(valid_steps, smoothed, 'r-', linewidth=2, alpha=0.7, label='Trend (Moving Average)')
            plt.legend()
        except Exception as e:
            print(f"Could not add trend line: {e}")
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Loss curve saved to {output_path}")
    plt.close()

def main():
    # Set these paths to match your configuration
    log_path = "./data/processed/logs"  
    output_path = "./data/processed/plots/training_loss.png"
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Extract loss values from logs
    steps, losses = extract_loss_from_logs(log_path)
    
    if not steps or not losses:
        print("No loss data found in logs.")
        return
    
    print(f"Found {len(steps)} loss data points")
    
    # Create and save the plot
    plot_training_loss(steps, losses, output_path)

if __name__ == "__main__":
    main()