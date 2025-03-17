"""
This script parses DGL-KE training logs and generates a plot of the training loss.
Can be run during training to visualize progress so far.
"""

import os
import re
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

def extract_loss_from_logs(log_path):
    """
    Extract training loss values from DGL-KE log files.
    Works for both completed and in-progress training.
    """
    # Updated patterns to match your log format
    loss_pattern = r"\[proc 0\]\[Train\]\((\d+)/\d+\) average loss: ([\d\.]+)"
    mrr_pattern = r"\[0\]Valid average MRR: ([\d\.]+)"
    
    steps = []
    losses = []
    valid_steps = []
    mrr_values = []
    
    # Find the training log file
    if os.path.isfile(log_path):
        log_file = log_path
    else:
        log_files = [f for f in os.listdir(log_path) if f.endswith('.log')]
        
        if not log_files:
            print(f"No log files found in {log_path}")
            return steps, losses, valid_steps, mrr_values
        
        # Use the most recent log file if there are multiple
        log_file = os.path.join(log_path, sorted(log_files)[-1])
    
    print(f"Parsing log file: {log_file}")
    
    with open(log_file, 'r') as f:
        log_content = f.read()
    
    # Find all loss entries with their steps
    loss_matches = re.findall(loss_pattern, log_content)
    
    if not loss_matches:
        print("No loss data found in logs. Check the log format.")
        return steps, losses, valid_steps, mrr_values
    
    # Extract validation MRR values
    current_step = 0
    for line in log_content.split('\n'):
        if '[proc 0][Train](' in line and 'average loss:' in line:
            match = re.search(r"\[proc 0\]\[Train\]\((\d+)/\d+\)", line)
            if match:
                current_step = int(match.group(1))
        
        # Check if this is an MRR line that comes after a step
        mrr_match = re.search(mrr_pattern, line)
        if mrr_match and current_step > 0:
            valid_steps.append(current_step)
            mrr_values.append(float(mrr_match.group(1)))
    
    # Process the loss matches
    for match in loss_matches:
        step = int(match[0])
        loss = float(match[1])
        steps.append(step)
        losses.append(loss)
    
    print(f"Found {len(steps)} loss data points (current step: {steps[-1]})")
    if mrr_values:
        print(f"Found {len(mrr_values)} validation MRR data points")
    
    return steps, losses, valid_steps, mrr_values

def plot_training_loss(steps, losses, valid_steps, mrr_values, output_path, total_steps=100000):
    """
    Create and save a plot of the training loss and validation MRR.
    Shows progress towards total_steps.
    """
    # Create figure with two subplots (training loss and validation MRR)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [2, 1]})
    
    # Plot training loss
    ax1.plot(steps, losses, marker='.', linestyle='-', markersize=2, color='blue', label='Training Loss')
    ax1.set_title('Training Loss over Time')
    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('Loss')
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Set x-axis to show full training range
    ax1.set_xlim([0, total_steps])
    
    # Add a vertical line showing current progress
    if steps:
        current_step = steps[-1]
        completion = current_step/total_steps*100
        ax1.axvline(x=current_step, color='g', linestyle='--', alpha=0.7, 
                   label=f'Current Step: {current_step}/{total_steps} ({completion:.1f}%)')
    
    # Add a smoothed trend line for loss
    if len(steps) > 5:
        try:
            window_size = max(5, len(steps) // 50)  # Use a reasonable window size
            smoothed = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')
            valid_x = steps[window_size-1:][:len(smoothed)]
            ax1.plot(valid_x, smoothed, 'r-', linewidth=2, alpha=0.7, label='Trend (Moving Average)')
        except Exception as e:
            print(f"Could not add trend line: {e}")
    
    ax1.legend()
    
    # Plot validation MRR
    if valid_steps and mrr_values:
        ax2.plot(valid_steps, mrr_values, marker='o', linestyle='-', markersize=4, color='purple', label='Validation MRR')
        ax2.set_title('Validation MRR over Time')
        ax2.set_xlabel('Training Steps')
        ax2.set_ylabel('MRR')
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.set_xlim([0, total_steps])
        
        # Add text with the best MRR value
        best_mrr = max(mrr_values)
        best_idx = mrr_values.index(best_mrr)
        best_step = valid_steps[best_idx]
        ax2.annotate(f'Best MRR: {best_mrr:.4f} at step {best_step}',
                    xy=(best_step, best_mrr), xytext=(best_step+5000, best_mrr),
                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3", color='black'),
                    bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.7))
    
    # Add current timestamp to the plot
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    plt.figtext(0.02, 0.02, f"Generated: {timestamp}", fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Loss curve saved to {output_path}")
    
    # Also save a timestamped version to keep history
    if steps:
        timestamped_path = output_path.replace('.png', f'_{steps[-1]}.png')
        plt.savefig(timestamped_path, dpi=300)
        print(f"Timestamped loss curve saved to {timestamped_path}")
    
    plt.close()

def main():
    # Set these paths to match your configuration
    log_path = "./data/processed/logs/training_20250316_234923.log"  # Direct path to your log file
    output_path = "./data/processed/plots/training_loss.png"
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Extract loss values from logs
    steps, losses, valid_steps, mrr_values = extract_loss_from_logs(log_path)
    
    if not steps or not losses:
        print("No loss data found in logs yet.")
        return
    
    # Create and save the plot
    plot_training_loss(steps, losses, valid_steps, mrr_values, output_path, total_steps=100000)

if __name__ == "__main__":
    main()