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
    
    plt.figure(figsize=(10, 6))
    plt.plot(steps, mrr_values, 'b-', linewidth=2, marker='o', markersize=5)
    
    # Add trend line
    z = np.polyfit(steps, mrr_values, 3)
    p = np.poly1d(z)
    plt.plot(steps, p(steps), 'r--', linewidth=1, alpha=0.7)
    
    plt.title('Mean Reciprocal Rank (MRR) Over Training Steps', fontsize=14)
    plt.xlabel('Training Steps', fontsize=12)
    plt.ylabel('MRR', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add annotations for initial and best values
    initial_mrr = mrr_values[0]
    best_mrr = max(mrr_values)
    best_step = steps[mrr_values.index(best_mrr)]
    
    plt.annotate(f'Initial: {initial_mrr:.4f}', 
                 xy=(steps[0], initial_mrr),
                 xytext=(steps[0] + 2000, initial_mrr - 0.02),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                 fontsize=10)
    
    plt.annotate(f'Best: {best_mrr:.4f}', 
                 xy=(best_step, best_mrr),
                 xytext=(best_step - 10000, best_mrr + 0.02),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                 fontsize=10)
    
    # Calculate improvement percentage
    improvement = ((best_mrr - initial_mrr) / initial_mrr) * 100
    plt.figtext(0.5, 0.01, f'Overall Improvement: {improvement:.2f}%', 
                ha='center', fontsize=12, bbox=dict(facecolor='lightgray', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"MRR plot saved to {output_path}")

if __name__ == "__main__":
    log_path = "./data/processed/logs/training_20250316_234923.log"
    output_path = "./data/processed/plots/mrr_over_training.png"
    plot_mrr(log_path, output_path)