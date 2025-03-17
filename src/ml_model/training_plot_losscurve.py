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
    pos_loss = [float(pl) for _, pl, _, _, _ in matches]
    neg_loss = [float(nl) for _, _, nl, _, _ in matches]
    total_loss = [float(tl) for _, _, _, tl, _ in matches]
    reg_loss = [float(rl) for _, _, _, _, rl in matches]
    
    return steps, pos_loss, neg_loss, total_loss, reg_loss

def plot_training_loss(log_path, output_path):
    steps, pos_loss, neg_loss, total_loss, reg_loss = extract_loss_metrics_from_log(log_path)
    
    plt.figure(figsize=(12, 8))
    
    # Plot all loss components
    plt.plot(steps, total_loss, 'b-', linewidth=2.5, label='Total Loss')
    plt.plot(steps, pos_loss, 'g-', linewidth=1.8, label='Positive Loss')
    plt.plot(steps, neg_loss, 'r-', linewidth=1.8, label='Negative Loss')
    plt.plot(steps, reg_loss, 'c-', linewidth=1.8, label='Regularization')
    
    plt.title('Training Loss Components Over Training Steps', fontsize=16)
    plt.xlabel('Training Steps', fontsize=14)
    plt.ylabel('Loss Value', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    # Add annotations for initial and final loss values
    initial_total = total_loss[0]
    final_total = total_loss[-1]
    
    plt.annotate(f'Initial: {initial_total:.3f}', 
                 xy=(steps[0], initial_total),
                 xytext=(steps[0] + 5000, initial_total + 0.5),
                 arrowprops=dict(facecolor='blue', shrink=0.05, width=1.5, headwidth=8),
                 fontsize=11)
    
    plt.annotate(f'Final: {final_total:.3f}', 
                 xy=(steps[-1], final_total),
                 xytext=(steps[-1] - 10000, final_total + 0.5),
                 arrowprops=dict(facecolor='blue', shrink=0.05, width=1.5, headwidth=8),
                 fontsize=11)
    
    # Calculate reduction percentage in total loss
    reduction = ((initial_total - final_total) / initial_total) * 100
    
    # Add detailed summary of loss reduction
    summary = (f"Loss Reduction:\n"
               f"Total Loss: {reduction:.1f}%\n"
               f"Pos. Loss: {((pos_loss[0] - pos_loss[-1]) / pos_loss[0] * 100):.1f}%\n"
               f"Neg. Loss: {((neg_loss[0] - neg_loss[-1]) / neg_loss[0] * 100):.1f}%")
    
    plt.figtext(0.15, 0.01, summary, fontsize=11, 
                bbox=dict(facecolor='lightblue', alpha=0.5))
    
    # Add vertical lines at significant training checkpoints
    for i in range(0, len(steps), 10):
        if i < len(steps):
            plt.axvline(x=steps[i], color='gray', linestyle=':', alpha=0.3)
    
    # Visualization enhancements
    plt.yscale('log')  # Log scale to better visualize different loss components
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    # Add a second plot showing the ratio of positive to negative loss
    ax2 = plt.twinx()
    ratio = [p/n for p, n in zip(pos_loss, neg_loss)]
    ax2.plot(steps, ratio, 'm--', linewidth=1.5, alpha=0.7, label='Pos/Neg Ratio')
    ax2.set_ylabel('Positive/Negative Loss Ratio', color='m', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='m')
    ax2.set_ylim(0, max(ratio)*1.2)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Training loss plot saved to {output_path}")
    
    # Create a second plot showing just the total loss with linear scale
    plt.figure(figsize=(10, 6))
    plt.plot(steps, total_loss, 'b-', linewidth=2.5)
    
    # Add trend line
    z = np.polyfit(steps, total_loss, 3)
    p = np.poly1d(z)
    plt.plot(steps, p(steps), 'r--', linewidth=1.5, alpha=0.7)
    
    plt.title('Total Training Loss', fontsize=14)
    plt.xlabel('Training Steps', fontsize=12)
    plt.ylabel('Loss Value', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add annotations
    plt.annotate(f'Initial: {initial_total:.3f}', 
                 xy=(steps[0], initial_total),
                 xytext=(steps[0] + 5000, initial_total + 0.2),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                 fontsize=10)
    
    plt.annotate(f'Final: {final_total:.3f}', 
                 xy=(steps[-1], final_total),
                 xytext=(steps[-1] - 10000, final_total + 0.2),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                 fontsize=10)
    
    linear_output_path = output_path.replace('.png', '_linear.png')
    plt.tight_layout()
    plt.savefig(linear_output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Linear scale training loss plot saved to {linear_output_path}")

if __name__ == "__main__":
    log_path = "./data/processed/logs/training_20250316_234923.log"
    output_path = "./data/processed/plots/training_loss_curves.png"
    plot_training_loss(log_path, output_path)