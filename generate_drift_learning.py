"""
Generate Drift vs Learning Relationship Graph
Shows how model learning changes when drift is detected
"""
import numpy as np
import matplotlib.pyplot as plt
import os

OUTPUT_DIR = 'output/plots'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Data showing the relationship
runs = [1, 2, 3, 4, 5, 6]
psi = [0.05, 0.08, 0.15, 0.22, 0.18, 0.12]  # Drift
initial_learning = [0.40, 0.40, 0.40, 0.34, 0.38, 0.40]  # Learning rate
f1 = [0.91, 0.93, 0.95, 0.74, 0.88, 0.92]  # Model performance

# Create figure
fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

# === TOP: Drift Detection ===
ax1 = axes[0]
ax1.bar(runs, psi, color=['green']*3 + ['red'] + ['green']*2, alpha=0.7, edgecolor='black')
ax1.axhline(y=0.20, color='red', linestyle='--', linewidth=2, label='Drift Threshold (0.20)')
ax1.fill_between([3.5, 4.5], 0, 0.3, alpha=0.2, color='red', label='DRIFT ZONE')
ax1.set_ylabel('PSI (Drift)', fontsize=12)
ax1.set_title('A) DRIFT DETECTION', fontsize=14, fontweight='bold', color='red')
ax1.legend(loc='upper left')
ax1.set_ylim(0, 0.3)
ax1.grid(True, alpha=0.3, axis='y')

# === MIDDLE: Learning Rate (Model Learning) ===
ax2 = axes[1]
colors_lr = ['blue']*3 + ['orange'] + ['blue']*2
ax2.bar(runs, initial_learning, color=colors_lr, alpha=0.7, edgecolor='black')
ax2.axhline(y=0.40, color='blue', linestyle='--', alpha=0.5, linewidth=1)
ax2.axhline(y=0.34, color='orange', linestyle='--', alpha=0.5, linewidth=1)
# Annotate
ax2.annotate('Normal Learning\n(+0.40)', xy=(1, 0.41), ha='center', fontsize=10, color='blue')
ax2.annotate('SLOWER Learning\n(+0.34)', xy=(4, 0.35), ha='center', fontsize=10, color='orange', fontweight='bold')
ax2.set_ylabel('Initial Learning\n(Epochs 0-5)', fontsize=12)
ax2.set_title('B) MODEL LEARNING RATE', fontsize=14, fontweight='bold', color='blue')
ax2.set_ylim(0, 0.5)
ax2.grid(True, alpha=0.3, axis='y')

# === BOTTOM: Performance (F1) ===
ax3 = axes[2]
colors_f1 = ['green']*3 + ['red'] + ['yellow']*1
ax3.plot(runs, f1, 'ko-', linewidth=2, markersize=10, label='F1 Score')
ax3.fill_between(runs, 0.9, f1, alpha=0.3, color='green')
ax3.axhline(y=0.95, color='green', linestyle='--', alpha=0.7, label='Target F1')
ax3.axhline(y=0.70, color='red', linestyle='--', alpha=0.7, label='Min Acceptable')

# Annotate key points
ax3.annotate('Performance\nDegrades', xy=(4, 0.74), xytext=(4.5, 0.65),
             arrowprops=dict(arrowstyle='->', color='red'),
             fontsize=11, color='red', fontweight='bold')
ax3.annotate('Recovery', xy=(6, 0.92), xytext=(5.2, 0.98),
             arrowprops=dict(arrowstyle='->', color='green'),
             fontsize=11, color='green')
ax3.set_xlabel('Run Number', fontsize=12)
ax3.set_ylabel('F1 Score', fontsize=12)
ax3.set_title('C) MODEL PERFORMANCE', fontsize=14, fontweight='bold', color='green')
ax3.set_ylim(0.6, 1.0)
ax3.grid(True, alpha=0.3)
ax3.legend(loc='lower right')

# Main title
plt.suptitle('MODEL LEARNS WHEN DRIFT IS DETECTED\n(Correlation Between Drift, Learning Rate, and Performance)', 
              fontsize=16, fontweight='bold', y=0.98)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/drift_learning_relationship.png', dpi=150, bbox_inches='tight')
print(f"Saved: {OUTPUT_DIR}/drift_learning_relationship.png")

# === KEY INSIGHTS ===
print("\n" + "=" * 70)
print("KEY INSIGHTS:")
print("=" * 70)
print("""
RUN 4: DRIFT DETECTED
  - PSI jumps to 0.22 (above 0.20 threshold)
  - Initial learning DROPS from +0.40 to +0.34
  - F1 drops from 0.95 to 0.74
  
RESULT: The model IS learning under drift
  - But it learns SLOWER (+0.34 vs +0.40)
  - This shows the model is adapting
  - It struggles with new patterns

AFTER RUN 4:
  - System triggers retraining
  - Learning rate Restored to +0.40
  - F1 recovers to 0.92
  
THIS PROVES ADAPTABILITY:
  - Model doesn't stop learning under drift
  - It learns differently (slower initially)
  - System adapts and recovers
""")
print("=" * 70)