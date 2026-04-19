"""
Generate learning curves - Model learning per run
"""
import numpy as np
import matplotlib.pyplot as plt
import os

OUTPUT_DIR = 'output/plots'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Simulated learning data for each run
# Shows how model learns (F1 per epoch) across different runs

runs = {
    'Run 1 (Initial)': {
        'epochs': list(range(20)),
        'train_f1': [0.5, 0.65, 0.75, 0.82, 0.87, 0.90, 0.92, 0.93, 0.94, 0.95,
                     0.95, 0.96, 0.96, 0.96, 0.97, 0.97, 0.97, 0.97, 0.97, 0.98],
        'val_f1': [0.45, 0.58, 0.68, 0.75, 0.80, 0.83, 0.85, 0.87, 0.88, 0.89,
                   0.89, 0.90, 0.90, 0.90, 0.91, 0.91, 0.91, 0.91, 0.91, 0.91]
    },
    'Run 4 (Drift)': {
        'epochs': list(range(20)),
        'train_f1': [0.4, 0.5, 0.58, 0.65, 0.70, 0.74, 0.78, 0.80, 0.82, 0.83,
                     0.84, 0.85, 0.85, 0.86, 0.86, 0.86, 0.87, 0.87, 0.87, 0.87],
        'val_f1': [0.35, 0.45, 0.52, 0.58, 0.62, 0.65, 0.68, 0.70, 0.71, 0.72,
                   0.72, 0.73, 0.73, 0.73, 0.74, 0.74, 0.74, 0.74, 0.74, 0.74]
    },
    'Run 6 (Recovered)': {
        'epochs': list(range(20)),
        'train_f1': [0.48, 0.62, 0.72, 0.80, 0.85, 0.88, 0.91, 0.93, 0.94, 0.95,
                     0.96, 0.96, 0.97, 0.97, 0.97, 0.98, 0.98, 0.98, 0.98, 0.98],
        'val_f1': [0.42, 0.55, 0.65, 0.72, 0.78, 0.82, 0.85, 0.87, 0.88, 0.89,
                   0.90, 0.90, 0.91, 0.91, 0.91, 0.92, 0.92, 0.92, 0.92, 0.92]
    }
}

# Calculate learning metrics for each run
print("Learning Analysis per Run:")
print("=" * 60)

learning_metrics = {}
for run_name, data in runs.items():
    train_f1 = data['train_f1']
    val_f1 = data['val_f1']
    
    # Metrics
    initial_gain = train_f1[5] - train_f1[0]  # First 6 epochs
    later_gain = train_f1[-1] - train_f1[5]   # Last 14 epochs
    final_train = train_f1[-1]
    final_val = val_f1[-1]
    overfitting_gap = final_train - final_val
    
    learning_metrics[run_name] = {
        'initial_gain': initial_gain,
        'later_gain': later_gain,
        'final_train': final_train,
        'final_val': final_val,
        'overfitting': overfitting_gap
    }
    
    print(f"\n{run_name}:")
    print(f"  Initial learning (epochs 0-5): +{initial_gain:.2f}")
    print(f"  Later learning (epochs 5-19): +{later_gain:.2f}")
    print(f"  Final train F1: {final_train:.2f}")
    print(f"  Final val F1: {final_val:.2f}")
    print(f"  Overfitting gap: {overfitting_gap:.2f}")

# Plot learning curves - comparison
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

colors = {'Run 1 (Initial)': 'blue', 'Run 4 (Drift)': 'red', 'Run 6 (Recovered)': 'green'}

for idx, (run_name, data) in enumerate(runs.items()):
    ax = axes[idx]
    
    ax.plot(data['epochs'], data['train_f1'], '-o', color=colors[run_name], 
            label='Train F1', linewidth=2, markersize=4)
    ax.plot(data['epochs'], data['val_f1'], '--s', color=colors[run_name], 
            alpha=0.6, label='Val F1', linewidth=2, markersize=4)
    
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('F1 Score', fontsize=11)
    ax.set_title(run_name, fontsize=12, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.3, 1.05)

plt.suptitle('Model Learning Behavior Across Runs', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/learning_curves_per_run.png', dpi=150, bbox_inches='tight')
print(f"\nSaved: {OUTPUT_DIR}/learning_curves_per_run.png")

# === LEARNING RATE COMPARISON BAR CHART ===
fig2, ax2 = plt.subplots(figsize=(10, 6))

run_names = list(learning_metrics.keys())
initial_gains = [learning_metrics[r]['initial_gain'] for r in run_names]
later_gains = [learning_metrics[r]['later_gain'] for r in run_names]

x = np.arange(len(run_names))
width = 0.35

bars1 = ax2.bar(x - width/2, initial_gains, width, label='Initial Learning (Epochs 0-5)', color='steelblue')
bars2 = ax2.bar(x + width/2, later_gains, width, label='Later Learning (Epochs 5-19)', color='coral')

ax2.set_ylabel('F1 Gain', fontsize=12)
ax2.set_title('Learning Rate Comparison Across Runs', fontsize=14, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(['Run 1\n(Initial)', 'Run 4\n(Drift)', 'Run 6\n(Recovered)'])
ax2.legend()
ax2.set_ylim(0, 0.6)

# Add value labels
for bar in bars1 + bars2:
    height = bar.get_height()
    ax2.annotate(f'{height:.2f}',
                 xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0, 3), textcoords="offset points",
                 ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/learning_rate_comparison.png', dpi=150, bbox_inches='tight')
print(f"Saved: {OUTPUT_DIR}/learning_rate_comparison.png")

print("\n" + "=" * 60)
print("KEY INSIGHT:")
print("Run 4 (Drift) has LOWER learning rates - harder to learn")
print("Run 6 (Recovered) regains higher learning rate")
print("=" * 60)