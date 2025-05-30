import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create outputs directory if it doesn't exist
os.makedirs('../outputs/temperature', exist_ok=True)

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def infonce_loss(pos_sim, neg_sims, temperature):
    """
    Calculate InfoNCE loss for cosine similarities

    Args:
        pos_sim: cosine similarity to positive sample [-1, 1]
        neg_sims: array of cosine similarities to negative samples [-1, 1]
        temperature: temperature parameter τ

    Returns:
        InfoNCE loss value
    """
    numerator = np.exp(pos_sim / temperature)
    denominator = np.sum(np.exp(neg_sims / temperature))
    return -np.log(numerator / denominator)


def generate_negative_samples(batch_size, difficulty='mixed'):
    """
    Generate negative samples with different difficulty levels

    Args:
        batch_size: number of negative samples (batch_size - 1)
        difficulty: 'easy', 'hard', or 'mixed'

    Returns:
        array of negative cosine similarities
    """
    n_negatives = batch_size - 1  # One sample is positive

    if difficulty == 'easy':
        # Mostly dissimilar negatives
        return np.random.uniform(-0.8, -0.2, n_negatives)
    elif difficulty == 'hard':
        # Mostly hard negatives (similar but wrong)
        return np.random.uniform(-0.1, 0.3, n_negatives)
    elif difficulty == 'mixed':
        # Realistic mix of easy and hard negatives
        easy_count = int(n_negatives * 0.6)  # 60% easy negatives
        hard_count = n_negatives - easy_count  # 40% hard negatives

        easy_negs = np.random.uniform(-0.8, -0.2, easy_count)
        hard_negs = np.random.uniform(-0.1, 0.3, hard_count)

        return np.concatenate([easy_negs, hard_negs])


# IMPORTANT: Negative sample similarities used throughout all plots
# These represent realistic negative cosine similarities in contrastive learning:
# -0.8: Very dissimilar (opposite-ish direction)
# -0.3: Moderately dissimilar
# -0.1: Slightly dissimilar
#  0.1: Slightly similar (but still negative sample)
#  0.2: Somewhat similar (but still negative sample)
neg_sims_fixed = np.array([-0.8, -0.3, -0.1, 0.1, 0.2])

print("=" * 80)
print("NEGATIVE SAMPLE SIMILARITIES USED IN ALL PLOTS")
print("=" * 80)
print(f"Negative similarities: {neg_sims_fixed}")
print()
print("📝 INTERPRETATION:")
print("   • -0.8: Very different samples (e.g., cat vs airplane)")
print("   • -0.3: Moderately different (e.g., different object categories)")
print("   • -0.1: Slightly different (e.g., different instances of similar objects)")
print("   •  0.1: Slightly similar but wrong (e.g., similar pose, different identity)")
print("   •  0.2: Somewhat similar but wrong (e.g., same category, different instance)")
print()
print("🎯 WHY THESE VALUES:")
print("   • Realistic range for learned embeddings in contrastive learning")
print("   • Mix of clearly negative (-0.8 to -0.1) and 'hard negatives' (0.1, 0.2)")
print("   • Hard negatives (positive cosine but wrong samples) are crucial for learning")
print("   • In practice, random negatives often have cosine similarities around 0 ± 0.3")
print("=" * 80)
print()

# PLOT 1: Loss vs Positive Cosine Similarity
print("Creating Plot 1: Loss vs Positive Cosine Similarity...")
fig1, ax1 = plt.subplots(1, 1, figsize=(10, 7))

pos_sims = np.linspace(-1, 1, 100)  # Full cosine similarity range
temperatures = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]

for temp in temperatures:
    losses = [infonce_loss(pos_sim, neg_sims_fixed, temp) for pos_sim in pos_sims]
    ax1.plot(pos_sims, losses, label=f'τ = {temp}', linewidth=3)

ax1.set_xlabel('Positive Cosine Similarity', fontsize=14)
ax1.set_ylabel('InfoNCE Loss', fontsize=14)
ax1.set_title('InfoNCE Loss vs Positive Cosine Similarity\n"How steep should the learning be?"',
              fontsize=16, fontweight='bold', pad=20)
ax1.legend(loc='upper right', fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 12)

# Add reference lines
ax1.axvline(x=0, color='gray', linestyle='--', alpha=0.6, linewidth=2, label='Orthogonal')
ax1.axvline(x=1, color='green', linestyle='--', alpha=0.6, linewidth=2, label='Identical')
ax1.axvline(x=-1, color='red', linestyle='--', alpha=0.6, linewidth=2, label='Opposite')

plt.tight_layout()
plt.savefig('../outputs/plot1_loss_vs_positive_similarity.png', dpi=300, bbox_inches='tight')
plt.close()

# PLOT 2: Probability Distribution
print("Creating Plot 2: Probability Distribution...")
fig2, ax2 = plt.subplots(1, 1, figsize=(12, 7))

# Realistic cosine similarities for demonstration
similarities = np.array([0.9, -0.3, -0.1, 0.1, 0.2, 0.4])  # First is positive
temp_values = [0.05, 0.1, 0.2, 0.5]

x_pos = np.arange(len(similarities))
width = 0.2

for i, temp in enumerate(temp_values):
    # Calculate softmax probabilities
    exp_sims = np.exp(similarities / temp)
    probabilities = exp_sims / np.sum(exp_sims)

    ax2.bar(x_pos + i * width, probabilities, width,
            label=f'τ = {temp}', alpha=0.8)

ax2.set_xlabel('Sample (Cosine Similarities)', fontsize=14)
ax2.set_ylabel('Softmax Probability', fontsize=14)
ax2.set_title('InfoNCE Probability Distribution\n"How confident should the model be?"',
              fontsize=16, fontweight='bold', pad=20)
ax2.legend(fontsize=12)

# Create labels with similarity values
labels = [f'Pos\n({similarities[i]:.1f})' if i == 0 else f'Neg{i}\n({similarities[i]:.1f})'
          for i in range(len(similarities))]
ax2.set_xticks(x_pos + width * 1.5)
ax2.set_xticklabels(labels, fontsize=11)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../outputs/plot2_probability_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# PLOT 3: Loss Heatmap
print("Creating Plot 3: Loss Landscape Heatmap...")
fig3, ax3 = plt.subplots(1, 1, figsize=(10, 8))

pos_sim_range = np.linspace(-1, 1, 50)  # Full cosine range
temp_range = np.linspace(0.01, 1.0, 50)
X, Y = np.meshgrid(pos_sim_range, temp_range)
Z = np.zeros_like(X)

for i, temp in enumerate(temp_range):
    for j, pos_sim in enumerate(pos_sim_range):
        Z[i, j] = infonce_loss(pos_sim, neg_sims_fixed, temp)

# Clip extreme values for better visualization
Z = np.clip(Z, 0, 15)

im = ax3.imshow(Z, extent=[-1, 1, 0.01, 1.0], aspect='auto', origin='lower',
                cmap='viridis', vmin=0, vmax=12)
cbar = plt.colorbar(im, ax=ax3, label='InfoNCE Loss', shrink=0.8)
cbar.ax.tick_params(labelsize=12)
ax3.set_xlabel('Positive Cosine Similarity', fontsize=14)
ax3.set_ylabel('Temperature τ', fontsize=14)
ax3.set_title('InfoNCE Loss Landscape\n"Where is the sweet spot?"',
              fontsize=16, fontweight='bold', pad=20)

# Add contour lines and reference lines
contours = ax3.contour(X, Y, Z, levels=[2, 4, 6, 8, 10], colors='white', alpha=0.7, linewidths=1.5)
ax3.clabel(contours, inline=True, fontsize=10, fmt='%.0f')
ax3.axvline(x=0, color='red', linestyle='--', alpha=0.8, linewidth=2, label='Orthogonal')
ax3.axvline(x=1, color='white', linestyle='--', alpha=0.8, linewidth=2, label='Perfect match')

plt.tight_layout()
plt.savefig('../outputs/plot3_loss_landscape_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

# PLOT 4: Batch Size Effect on InfoNCE Loss
print("Creating Plot 4: Batch Size Effect on InfoNCE Loss...")
fig4, ((ax4a, ax4b), (ax4c, ax4d)) = plt.subplots(2, 2, figsize=(15, 12))

# Set random seed for reproducible results
np.random.seed(42)

# 4A: Loss vs Batch Size for different positive similarities
batch_sizes = [4, 8, 16, 32, 64, 128, 256, 512]
pos_similarities = [0.3, 0.5, 0.7, 0.9]
temperature = 0.1
n_trials = 10  # Average over multiple trials for stability

for pos_sim in pos_similarities:
    avg_losses = []
    std_losses = []

    for batch_size in batch_sizes:
        trial_losses = []
        for _ in range(n_trials):
            neg_sims = generate_negative_samples(batch_size, difficulty='mixed')
            loss = infonce_loss(pos_sim, neg_sims, temperature)
            trial_losses.append(loss)

        avg_losses.append(np.mean(trial_losses))
        std_losses.append(np.std(trial_losses))

    ax4a.errorbar(batch_sizes, avg_losses, yerr=std_losses,
                  label=f'Pos sim = {pos_sim}', marker='o', linewidth=2, markersize=6)

ax4a.set_xscale('log', base=2)
ax4a.set_xlabel('Batch Size', fontsize=12)
ax4a.set_ylabel('InfoNCE Loss', fontsize=12)
ax4a.set_title('Loss vs Batch Size\n(Different positive similarities)', fontsize=14, fontweight='bold')
ax4a.legend(fontsize=10)
ax4a.grid(True, alpha=0.3)

# 4B: Loss vs Batch Size for different temperatures
temperatures = [0.05, 0.1, 0.2, 0.5]
positive_sim = 0.7

for temp in temperatures:
    avg_losses = []
    std_losses = []

    for batch_size in batch_sizes:
        trial_losses = []
        for _ in range(n_trials):
            neg_sims = generate_negative_samples(batch_size, difficulty='mixed')
            loss = infonce_loss(positive_sim, neg_sims, temp)
            trial_losses.append(loss)

        avg_losses.append(np.mean(trial_losses))
        std_losses.append(np.std(trial_losses))

    ax4b.errorbar(batch_sizes, avg_losses, yerr=std_losses,
                  label=f'τ = {temp}', marker='s', linewidth=2, markersize=6)

ax4b.set_xscale('log', base=2)
ax4b.set_xlabel('Batch Size', fontsize=12)
ax4b.set_ylabel('InfoNCE Loss', fontsize=12)
ax4b.set_title('Loss vs Batch Size\n(Different temperatures)', fontsize=14, fontweight='bold')
ax4b.legend(fontsize=10)
ax4b.grid(True, alpha=0.3)

# 4C: Loss vs Batch Size for different negative difficulties
difficulties = ['easy', 'mixed', 'hard']
colors = ['green', 'blue', 'red']
positive_sim = 0.7
temperature = 0.1

for difficulty, color in zip(difficulties, colors):
    avg_losses = []
    std_losses = []

    for batch_size in batch_sizes:
        trial_losses = []
        for _ in range(n_trials):
            neg_sims = generate_negative_samples(batch_size, difficulty=difficulty)
            loss = infonce_loss(positive_sim, neg_sims, temperature)
            trial_losses.append(loss)

        avg_losses.append(np.mean(trial_losses))
        std_losses.append(np.std(trial_losses))

    ax4c.errorbar(batch_sizes, avg_losses, yerr=std_losses,
                  label=f'{difficulty.capitalize()} negatives', marker='^',
                  linewidth=2, markersize=6, color=color)

ax4c.set_xscale('log', base=2)
ax4c.set_xlabel('Batch Size', fontsize=12)
ax4c.set_ylabel('InfoNCE Loss', fontsize=12)
ax4c.set_title('Loss vs Batch Size\n(Different negative difficulties)', fontsize=14, fontweight='bold')
ax4c.legend(fontsize=10)
ax4c.grid(True, alpha=0.3)

# 4D: Variance in loss vs Batch Size
positive_sim = 0.7
temperature = 0.1
difficulties = ['easy', 'mixed', 'hard']

for difficulty, color in zip(difficulties, colors):
    variances = []

    for batch_size in batch_sizes:
        trial_losses = []
        for _ in range(50):  # More trials for variance estimation
            neg_sims = generate_negative_samples(batch_size, difficulty=difficulty)
            loss = infonce_loss(positive_sim, neg_sims, temperature)
            trial_losses.append(loss)

        variances.append(np.var(trial_losses))

    ax4d.plot(batch_sizes, variances, label=f'{difficulty.capitalize()} negatives',
              marker='D', linewidth=2, markersize=6, color=color)

ax4d.set_xscale('log', base=2)
ax4d.set_yscale('log')
ax4d.set_xlabel('Batch Size', fontsize=12)
ax4d.set_ylabel('Loss Variance', fontsize=12)
ax4d.set_title('Loss Variance vs Batch Size\n(Training stability)', fontsize=14, fontweight='bold')
ax4d.legend(fontsize=10)
ax4d.grid(True, alpha=0.3)

plt.suptitle('Batch Size Effects on InfoNCE Loss\n"Why bigger batches often work better"',
             fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout()
plt.subplots_adjust(top=0.93)
plt.savefig('../outputs/plot4_batch_size_effects.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n✅ All plots saved successfully to ./outputs/")
print("\n📁 Files created:")
print("   • plot1_loss_vs_positive_similarity.png")
print("   • plot2_probability_distribution.png")
print("   • plot3_loss_landscape_heatmap.png")
print("   • plot4_batch_size_effects.png")

# Create a summary figure with all negative similarities info
print("\nCreating summary of negative similarities used...")
fig_summary, ax_summary = plt.subplots(1, 1, figsize=(12, 8))

# Create a visual representation of the negative similarities
neg_labels = ['Neg 1', 'Neg 2', 'Neg 3', 'Neg 4', 'Neg 5']
colors = ['darkred', 'red', 'orange', 'lightblue', 'blue']

bars = ax_summary.bar(neg_labels, neg_sims_fixed, color=colors, alpha=0.8, edgecolor='black', linewidth=2)

# Add value labels on bars
for bar, sim in zip(bars, neg_sims_fixed):
    height = bar.get_height()
    ax_summary.text(bar.get_x() + bar.get_width() / 2., height + 0.02 if height > 0 else height - 0.05,
                    f'{sim:.1f}', ha='center', va='bottom' if height > 0 else 'top',
                    fontsize=14, fontweight='bold')

ax_summary.set_ylabel('Cosine Similarity', fontsize=14)
ax_summary.set_title('Negative Sample Similarities Used in InfoNCE Analysis\n' +
                     f'Values: {neg_sims_fixed}', fontsize=16, fontweight='bold', pad=20)
ax_summary.grid(True, alpha=0.3)
ax_summary.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=2)

# Add interpretation text
interpretation_text = """
Interpretation of Negative Similarities:
• -0.8: Very dissimilar samples (e.g., cat vs airplane)
• -0.3: Moderately different (e.g., different object categories)  
• -0.1: Slightly different (e.g., different instances of similar objects)
•  0.1: Slightly similar but wrong (e.g., similar pose, different identity)
•  0.2: Somewhat similar but wrong (e.g., same category, different instance)

Why these values matter:
• Realistic range for learned embeddings in contrastive learning
• Mix of clearly negative (-0.8 to -0.1) and 'hard negatives' (0.1, 0.2)
• Hard negatives are crucial for learning discriminative representations
"""

ax_summary.text(0.02, 0.02, interpretation_text, transform=ax_summary.transAxes, fontsize=11,
                verticalalignment='bottom',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow', alpha=0.9))

plt.tight_layout()
plt.savefig('../outputs/negative_similarities_explanation.png', dpi=300, bbox_inches='tight')
plt.close()

print("   • negative_similarities_explanation.png")

# Print insights about batch size effects
print("\n" + "=" * 80)
print("🔍 KEY INSIGHTS FROM BATCH SIZE ANALYSIS")
print("=" * 80)
print("📈 BATCH SIZE EFFECTS ON INFONCE:")
print("   • Larger batches → More negative samples → Generally lower loss")
print("   • Diminishing returns: Loss reduction slows at very large batch sizes")
print("   • Variance decreases with larger batches → More stable training")
print("   • Hard negatives make batch size effects more pronounced")
print()
print("🎯 PRACTICAL IMPLICATIONS:")
print("   • Start with batch sizes of 64-256 for good performance")
print("   • Very large batches (512+) may not justify the memory cost")
print("   • Hard negative mining becomes more effective with larger batches")
print("   • Temperature tuning becomes more critical with smaller batches")
print()
print("⚖️  TRADE-OFFS:")
print("   • Larger batches: Better gradients, more memory, slower iterations")
print("   • Smaller batches: Less memory, noisier gradients, faster iterations")
print("   • Sweet spot often around 128-256 for most contrastive learning tasks")
print("=" * 80)
print("\n🎯 Key Point: The choice of batch size significantly affects")
print("   InfoNCE loss behavior, training stability, and convergence!")