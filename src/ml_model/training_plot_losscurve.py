import re
import matplotlib.pyplot as plt
import numpy as np

def extract_loss_metrics_from_log(log_path):
    with open(log_path, 'r') as f:
        log_content = f.read()
    
    # Extract training steps and different loss values
    pattern = r'\[proc 0\]\[Train\]\((\d+)/100000\) average pos_loss: ([\d\.]+).*?average neg_loss: ([\d\.]+).*?average loss: ([\d\.]+).*?average regularization: ([\d\.]+)'
    matches = re.findall(pattern, log_content, re.DOTALL)
    
    steps = [int(step) for step, _, _, _, _ in matches]
    total_loss = [float(tl) for _, _, _, tl, _ in matches]
    
    return steps, total_loss

def plot_training_loss(log_path, output_path):
    steps, total_loss = extract_loss_metrics_from_log(log_path)
    
    plt.figure(figsize=(12, 8))
    
    # Plot only total loss
    plt.plot(steps, total_loss, 'b-', linewidth=2.5, label='Total Loss')
    
    plt.title('Total Training Loss', fontsize=16)
    plt.xlabel('Training Steps', fontsize=14)
    plt.ylabel('Loss Value', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add annotations for initial and final loss values
    initial_total = total_loss[0]
    final_total = total_loss[-1]
    
    plt.annotate(f'Initial: {initial_total:.3f}', 
                 xy=(steps[0], initial_total),
                 xytext=(steps[0] + 5000, initial_total * 0.9),
                 arrowprops=dict(facecolor='blue', shrink=0.05, width=1.5, headwidth=8),
                 fontsize=11)
    
    plt.annotate(f'Final: {final_total:.3f}', 
                 xy=(steps[-1], final_total),
                 xytext=(steps[-1] - 15000, final_total * 1.1),
                 arrowprops=dict(facecolor='blue', shrink=0.05, width=1.5, headwidth=8),
                 fontsize=11)
    
    # Calculate reduction percentage in total loss
    reduction = ((initial_total - final_total) / initial_total) * 100
    
    # Add summary of loss reduction
    summary = f"Loss Reduction: {reduction:.1f}%"
    
    plt.figtext(0.15, 0.01, summary, fontsize=11, 
                bbox=dict(facecolor='lightblue', alpha=0.5))
    
    # Add trend line
    z = np.polyfit(steps, total_loss, 3)
    p = np.poly1d(z)
    plt.plot(steps, p(steps), 'r--', linewidth=1.5, alpha=0.7, label='Trend')
    
    # Add legend
    plt.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Training loss plot saved to {output_path}")

if __name__ == "__main__":
    log_path = "./data/processed/logs/training_20250316_234923.log"
    output_path = "./data/processed/plots/total_loss_curve.png"
    plot_training_loss(log_path, output_path)