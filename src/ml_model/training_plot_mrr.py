import re
import matplotlib.pyplot as plt
import numpy as np

def extract_metrics_from_log(log_path):
    with open(log_path, 'r') as f:
        log_content = f.read()
    
    # Extract steps and MRR values
    pattern = r'\[proc 0\]\[Train\]\((\d+)/100000\).*?\[0\]Valid average MRR: ([\d\.]+)'
    matches = re.findall(pattern, log_content, re.DOTALL)
    
    steps = [int(step) for step, _ in matches]
    mrr_values = [float(mrr) for _, mrr in matches]
    
    return steps, mrr_values

def plot_mrr(log_path, output_path):
    steps, mrr_values = extract_metrics_from_log(log_path)
    
    # Create figure and axes for better control
    fig, ax = plt.subplots(figsize=(12, 7))
    
    ax.plot(steps, mrr_values, 'b-', linewidth=2, marker='o', markersize=5)
    
    # Add trend line
    z = np.polyfit(steps, mrr_values, 3)
    p = np.poly1d(z)
    ax.plot(steps, p(steps), 'r--', linewidth=1, alpha=0.7)
    
    ax.set_title('Mean Reciprocal Rank (MRR) Over Training Steps', fontsize=14)
    ax.set_xlabel('Training Steps', fontsize=12)
    ax.set_ylabel('MRR', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add annotations for initial and best values
    initial_mrr = mrr_values[0]
    best_mrr = max(mrr_values)
    best_step = steps[mrr_values.index(best_mrr)]
    
    ax.annotate(f'Initial: {initial_mrr:.4f}', 
               xy=(steps[0], initial_mrr),
               xytext=(steps[0] + 2000, initial_mrr - 0.02),
               arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
               fontsize=10)
    
    ax.annotate(f'Best: {best_mrr:.4f}', 
               xy=(best_step, best_mrr),
               xytext=(best_step - 10000, best_mrr + 0.02),
               arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
               fontsize=10)
    
    # Calculate improvement percentage
    improvement = ((best_mrr - initial_mrr) / initial_mrr) * 100
    
    # Get the current axis limits
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    
    # Position the text box in the center-bottom portion of the plot,
    # Ensuring it's above the x-axis but below the data
    text_x = (xmin + xmax) / 2  # Center horizontally
    text_y = ymin + (ymax - ymin) * 0.15  # 15% up from the bottom
    
    # Draw the text box with the improvement percentage
    ax.text(text_x, text_y, f'Overall Improvement: {improvement:.2f}%', 
            fontsize=12, 
            bbox=dict(facecolor='lightgray', alpha=0.5, boxstyle='round,pad=0.5'),
            ha='center')  # Center horizontally
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"MRR plot saved to {output_path}")

if __name__ == "__main__":
    log_path = "./data/processed/logs/training_20250316_234923.log"
    output_path = "./data/processed/plots/mrr_over_training.png"
    plot_mrr(log_path, output_path)