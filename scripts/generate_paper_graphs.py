#!/usr/bin/env python3
"""
Generate graphs for paper results
"""
import matplotlib.pyplot as plt
import numpy as np

# Results data
phases = ['Initial', 'After Drift', 'After Retrain']
f1_scores = [0.9877, 0.9726, 0.9929]
roc_scores = [0.9869, 0.9271, 0.9943]

drift_metrics = ['PSI', 'KS']
drift_values = [8.09, 0.50]
drift_thresholds = [0.20, 0.05]

# Create figure
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot 1: F1 and ROC-AUC comparison
ax1 = axes[0]
x = np.arange(len(phases))
width = 0.35

bars1 = ax1.bar(x - width/2, f1_scores, width, label='F1-Score', color='steelblue')
bars2 = ax1.bar(x + width/2, roc_scores, width, label='ROC-AUC', color='coral')

ax1.set_ylabel('Score')
ax1.set_title('Model Performance: Drift Adaptation')
ax1.set_xticks(x)
ax1.set_xticklabels(phases)
ax1.legend()
ax1.set_ylim(0.9, 1.0)

# Add value labels
for bar in bars1:
    height = bar.get_height()
    ax1.annotate(f'{height:.3f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontsize=9)
for bar in bars2:
    height = bar.get_height()
    ax1.annotate(f'{height:.3f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontsize=9)

# Plot 2: Drift Detection
ax2 = axes[1]
colors = ['red' if v > t else 'green' for v, t in zip(drift_values, drift_thresholds)]
bars = ax2.bar(drift_metrics, drift_values, color=colors)
ax2.axhline(y=0.20, color='red', linestyle='--', label='PSI threshold (0.20)')
ax2.axhline(y=0.05, color='orange', linestyle='--', label='KS threshold (0.05)')
ax2.set_ylabel('Value')
ax2.set_title('Drift Detection Metrics')
ax2.legend()

# Add value labels
for bar, val in zip(bars, drift_values):
    ax2.annotate(f'{val:.2f}',
                xy=(bar.get_x() + bar.get_width() / 2, val),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontsize=10)

# Plot 3: Performance degradation and recovery
ax3 = axes[2]
degradation = [f1_scores[0] - f1_scores[1], roc_scores[0] - roc_scores[1]]
recovery = [f1_scores[2] - f1_scores[1], roc_scores[2] - roc_scores[1]]

x = np.arange(2)
bars3 = ax3.bar(x - width/2, degradation, width, label='Degradation', color='red', alpha=0.7)
bars4 = ax3.bar(x + width/2, recovery, width, label='Recovery', color='green', alpha=0.7)

ax3.set_ylabel('Change')
ax3.set_title('Degradation vs Recovery')
ax3.set_xticks(x)
ax3.set_xticklabels(['F1', 'ROC-AUC'])
ax3.legend()
ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

# Add value labels
for bar in bars3:
    height = bar.get_height()
    ax3.annotate(f'{height:+.3f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3 if height > 0 else -12), textcoords="offset points",
                ha='center', va='bottom', fontsize=9)
for bar in bars4:
    height = bar.get_height()
    ax3.annotate(f'{height:+.3f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3 if height > 0 else -12), textcoords="offset points",
                ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('output/plots/drift_adaptation_results.png', dpi=300, bbox_inches='tight')
plt.close()

print("Generated: output/plots/drift_adaptation_results.png")