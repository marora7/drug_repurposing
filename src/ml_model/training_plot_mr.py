import re
import matplotlib.pyplot as plt
import numpy as np

def extract_mean_rank_from_log(log_path):
    with open(log_path, 'r') as f:
        log_content = f.read()
    
    # Extract training steps and Mean Rank values
    pattern = r'\[proc 0\]\[Train\]\((\d+)/100000\).*?\[0\]Valid average MR: ([\d\.]+)'
    matches = re.findall(pattern, log_content, re.DOTALL)
    
    steps = [int(step) for step, _ in matches]
    mean_ranks = [float(mr) for _, mr in matches]
    
    return steps, mean_ranks

def plot_mean_rank(log_path, output_path):
    steps, mean_ranks = extract_mean_rank_from_log(log_path)
    
    # Create figure and axes for better control
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # For Mean Rank, lower is better, so using a different color scheme
    ax.plot(steps, mean_ranks, 'purple', linewidth=2, marker='o', markersize=5)
    
    # Add trend line
    z = np.polyfit(steps, mean_ranks, 3)
    p = np.poly1d(z)
    ax.plot(steps, p(steps), 'magenta', linestyle='--', linewidth=1, alpha=0.7)
    
    ax.set_title('Mean Rank Over Training Steps (Lower is Better)', fontsize=14)
    ax.set_xlabel('Training Steps', fontsize=12)
    ax.set_ylabel('Mean Rank', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add annotations for initial and best values
    initial_mr = mean_ranks[0]
    best_mr = min(mean_ranks)
    best_step = steps[mean_ranks.index(best_mr)]
    
    ax.annotate(f'Initial: {initial_mr:.2f}', 
                xy=(steps[0], initial_mr),
                xytext=(steps[0] + 5000, initial_mr + 50),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                fontsize=10)
    
    ax.annotate(f'Best: {best_mr:.2f}', 
                xy=(best_step, best_mr),
                xytext=(best_step - 10000, best_mr + 20),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                fontsize=10)
    
    # Calculate improvement percentage (for Mean Rank, reduction is improvement)
    improvement = ((initial_mr - best_mr) / initial_mr) * 100
    
    # Add shading to highlight improvement area
    ax.fill_between(steps, initial_mr, mean_ranks, alpha=0.2, color='purple')
    
    # Get the current axis limits
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    
    # Add summary text box - positioned in the right side but lower to avoid the line
    # Get the y-value of the curve at the position where we want to place the text
    curve_position_x = xmax * 0.7
    # Find the nearest x value in steps to our desired position
    nearest_idx = min(range(len(steps)), key=lambda i: abs(steps[i] - curve_position_x))
    curve_position_y = mean_ranks[nearest_idx]
    
    # Use a completely different approach - position the text in the bottom right portion of the plot
    # Regardless of the curve position, place it at 20% of y-axis height from the bottom
    text_x = xmax * 0.75  # 75% from the left edge
    text_y = ymin + (ymax - ymin) * 0.2  # Fixed at 20% of total height from bottom
    
    ax.text(text_x, text_y, f'Mean Rank Reduction: {improvement:.2f}%', 
            fontsize=12, 
            bbox=dict(facecolor='lavender', alpha=0.8, boxstyle='round,pad=0.5'),
            ha='center')  # Horizontally center the text in the box
    
    # Adjust y-axis to start from 0 to better visualize the magnitude of improvement
    ax.set_ylim(0, max(mean_ranks) * 1.1)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Mean Rank plot saved to {output_path}")

if __name__ == "__main__":
    log_path = "./data/processed/logs/training_20250316_234923.log"
    output_path = "./data/processed/plots/mean_rank_over_training.png"
    plot_mean_rank(log_path, output_path)