import re
import matplotlib.pyplot as plt
import numpy as np

def extract_hits_metrics_from_log(log_path):
    with open(log_path, 'r') as f:
        log_content = f.read()
    
    # Extract steps and Hits@K values together to ensure proper alignment
    pattern = r'\[proc 0\]\[Train\]\((\d+)/100000\).*?\[0\]Valid average HITS@1: ([\d\.]+).*?\[0\]Valid average HITS@3: ([\d\.]+).*?\[0\]Valid average HITS@10: ([\d\.]+)'
    matches = re.findall(pattern, log_content, re.DOTALL)
    
    steps = [int(step) for step, _, _, _ in matches]
    hits1_values = [float(hits1) for _, hits1, _, _ in matches]
    hits3_values = [float(hits3) for _, _, hits3, _ in matches]
    hits10_values = [float(hits10) for _, _, _, hits10 in matches]
    
    return steps, hits1_values, hits3_values, hits10_values

def plot_hits_at_k(log_path, output_path):
    steps, hits1_values, hits3_values, hits10_values = extract_hits_metrics_from_log(log_path)
    
    plt.figure(figsize=(12, 7))
    
    # Plot all Hits@K metrics
    plt.plot(steps, hits1_values, 'b-', linewidth=2, marker='o', markersize=4, label='Hits@1')
    plt.plot(steps, hits3_values, 'g-', linewidth=2, marker='s', markersize=4, label='Hits@3')
    plt.plot(steps, hits10_values, 'r-', linewidth=2, marker='^', markersize=4, label='Hits@10')
    
    plt.title('Hits@K Metrics Over Training Steps', fontsize=15)
    plt.xlabel('Training Steps', fontsize=13)
    plt.ylabel('Hits@K Value', fontsize=13)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    # Add annotations for best values
    best_hits1 = max(hits1_values)
    best_hits3 = max(hits3_values)
    best_hits10 = max(hits10_values)
    
    best_step1 = steps[hits1_values.index(best_hits1)]
    best_step3 = steps[hits3_values.index(best_hits3)]
    best_step10 = steps[hits10_values.index(best_hits10)]
    
    plt.annotate(f'Best Hits@1: {best_hits1:.4f}', 
                 xy=(best_step1, best_hits1),
                 xytext=(best_step1 - 8000, best_hits1 + 0.03),
                 arrowprops=dict(facecolor='blue', shrink=0.05, width=1, headwidth=6),
                 fontsize=9, color='blue')
    
    plt.annotate(f'Best Hits@3: {best_hits3:.4f}', 
                 xy=(best_step3, best_hits3),
                 xytext=(best_step3 - 5000, best_hits3 + 0.04),
                 arrowprops=dict(facecolor='green', shrink=0.05, width=1, headwidth=6),
                 fontsize=9, color='green')
    
    plt.annotate(f'Best Hits@10: {best_hits10:.4f}', 
                 xy=(best_step10, best_hits10),
                 xytext=(best_step10 - 15000, best_hits10 - 0.05),
                 arrowprops=dict(facecolor='red', shrink=0.05, width=1, headwidth=6),
                 fontsize=9, color='red')
    
    # Summary text box
    improvement1 = ((best_hits1 - hits1_values[0]) / hits1_values[0]) * 100
    improvement3 = ((best_hits3 - hits3_values[0]) / hits3_values[0]) * 100
    improvement10 = ((best_hits10 - hits10_values[0]) / hits10_values[0]) * 100
    
    summary = (f"Improvements:\n"
               f"Hits@1: {improvement1:.1f}%\n"
               f"Hits@3: {improvement3:.1f}%\n"
               f"Hits@10: {improvement10:.1f}%")
    
    plt.figtext(0.15, 0.01, summary, fontsize=11, 
                bbox=dict(facecolor='lightgray', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Hits@K plot saved to {output_path}")

if __name__ == "__main__":
    log_path = "./data/processed/logs/training_20250316_234923.log"
    output_path = "./data/processed/plots/hits_at_k_over_training.png"
    plot_hits_at_k(log_path, output_path)