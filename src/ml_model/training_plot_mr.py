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
    
    plt.figure(figsize=(10, 6))
    
    # For Mean Rank, lower is better, so using a different color scheme
    plt.plot(steps, mean_ranks, 'purple', linewidth=2, marker='o', markersize=5)
    
    # Add trend line
    z = np.polyfit(steps, mean_ranks, 3)
    p = np.poly1d(z)
    plt.plot(steps, p(steps), 'magenta', linestyle='--', linewidth=1, alpha=0.7)
    
    plt.title('Mean Rank Over Training Steps (Lower is Better)', fontsize=14)
    plt.xlabel('Training Steps', fontsize=12)
    plt.ylabel('Mean Rank', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add annotations for initial and best values
    initial_mr = mean_ranks[0]
    best_mr = min(mean_ranks)
    best_step = steps[mean_ranks.index(best_mr)]
    
    plt.annotate(f'Initial: {initial_mr:.2f}', 
                 xy=(steps[0], initial_mr),
                 xytext=(steps[0] + 5000, initial_mr + 50),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                 fontsize=10)
    
    plt.annotate(f'Best: {best_mr:.2f}', 
                 xy=(best_step, best_mr),
                 xytext=(best_step - 10000, best_mr + 20),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                 fontsize=10)
    
    # Calculate improvement percentage (for Mean Rank, reduction is improvement)
    improvement = ((initial_mr - best_mr) / initial_mr) * 100
    
    # Add shading to highlight improvement area
    plt.fill_between(steps, initial_mr, mean_ranks, alpha=0.2, color='purple')
    
    # Add summary text
    plt.figtext(0.5, 0.01, f'Mean Rank Reduction: {improvement:.2f}%', 
                ha='center', fontsize=12, bbox=dict(facecolor='lavender', alpha=0.5))
    
    # Adjust y-axis to start from 0 to better visualize the magnitude of improvement
    plt.ylim(0, max(mean_ranks) * 1.1)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Mean Rank plot saved to {output_path}")

if __name__ == "__main__":
    log_path = "./data/processed/logs/training_20250316_234923.log"
    output_path = "./data/processed/plots/mean_rank_over_training.png"
    plot_mean_rank(log_path, output_path)