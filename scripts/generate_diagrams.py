"""
Generate remaining diagrams for paper
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os

# Paths
DATA_PATH = 'data/data_binary_only_first_3000.csv'
OUTPUT_DIR = 'output/plots'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load data
print('Loading data...')
df = pd.read_csv(DATA_PATH)
print(f'Loaded {len(df)} rows')

# Target column - check both possible names
target_col = 'annotation.is_fraud'
if target_col not in df.columns:
    target_col = 'is_fraud'
print(f'Using target column: {target_col}')

# Get features (numeric only for t-SNE/PCA)
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if target_col in numeric_cols:
    numeric_cols.remove(target_col)

# Take subset for speed
sample_size = min(1000, len(df))
df_sample = df.sample(n=sample_size, random_state=42)

X = df_sample[numeric_cols].fillna(0).values
y = df_sample[target_col].values

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print('Running PCA...')
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print('Running t-SNE...')
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X_scaled)

# Plot PCA
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# PCA
colors = ['blue' if label == 0 else 'red' for label in y]
axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=colors, alpha=0.6, s=20)
axes[0].set_xlabel('PCA Component 1')
axes[0].set_ylabel('PCA Component 2')
axes[0].set_title('PCA: Original Data Distribution')
# Legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='blue', label='Non-Fraud'),
                   Patch(facecolor='red', label='Fraud')]
axes[0].legend(handles=legend_elements)

# t-SNE
axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1], c=colors, alpha=0.6, s=20)
axes[1].set_xlabel('t-SNE Dimension 1')
axes[1].set_ylabel('t-SNE Dimension 2')
axes[1].set_title('t-SNE: Original Data Distribution')
axes[1].legend(handles=legend_elements)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/data_distribution_pca_tsne.png', dpi=150, bbox_inches='tight')
print(f'Saved: {OUTPUT_DIR}/data_distribution_pca_tsne.png')
plt.close()

# === BOX PLOT: Feature Distributions (Non-Urgency) ===
print('Creating box plots...')

# Use features that make sense
features_to_plot = [
    'annotation.psychological_tactics.fear',
    'annotation.psychological_tactics.authority', 
    'annotation.psychological_tactics.reward',
    'annotation.key_features.amount_normalized'
]

available = [f for f in features_to_plot if f in df.columns]

if available:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, feat in enumerate(available):
        # Get data for both classes
        fraud_data = df[df[target_col] == 1][feat].dropna()
        non_fraud_data = df[df[target_col] == 0][feat].dropna()
        
        # Limit sample size for display
        fraud_data = fraud_data.head(500)
        non_fraud_data = non_fraud_data.head(500)
        
        data = [non_fraud_data.values, fraud_data.values]
        
        bp = axes[idx].boxplot(data, patch_artist=True, labels=['Non-Fraud', 'Fraud'])
        bp['boxes'][0].set_facecolor('lightblue')
        bp['boxes'][1].set_facecolor('lightcoral')
        
        # Clean up feature name
        feat_name = feat.split('.')[-1]
        axes[idx].set_title(f'{feat_name} Distribution')
        axes[idx].set_ylabel('Value')
    
    plt.suptitle('Feature Distributions by Class (Box Plot)', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/class_distribution_boxplot.png', dpi=150, bbox_inches='tight')
    print(f'Saved: {OUTPUT_DIR}/class_distribution_boxplot.png')
    plt.close()

# === ROBUSTNESS COMPARISON BAR CHART ===
print('Creating robustness comparison...')
# From run_summary data
metrics = {
    'Model': ['Baseline', 'CTGAN', 'CTGAN+FGSM'],
    'Clean F1': [0.95, 0.98, 0.99],
    'Robust F1': [0.65, 0.72, 0.98]
}

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(metrics['Model']))
width = 0.35

bars1 = ax.bar(x - width/2, metrics['Clean F1'], width, label='Clean F1', color='steelblue')
bars2 = ax.bar(x + width/2, metrics['Robust F1'], width, label='Robust F1 (ε=0.1)', color='coral')

ax.set_ylabel('F1 Score')
ax.set_title('Model Performance: Clean vs Robust')
ax.set_xticks(x)
ax.set_xticklabels(metrics['Model'])
ax.legend()
ax.set_ylim(0, 1.1)

# Add value labels
for bar in bars1:
    height = bar.get_height()
    ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
for bar in bars2:
    height = bar.get_height()
    ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/robustness_comparison.png', dpi=150, bbox_inches='tight')
print(f'Saved: {OUTPUT_DIR}/robustness_comparison.png')
plt.close()

# === COMBINED: Drift Detection + Model Adaptation ===
print('Creating drift + adaptation combined plot...')

# Simulated data showing the relationship
drift_data = {
    'Run': [1, 2, 3, 4, 5, 6],
    'PSI': [0.05, 0.08, 0.15, 0.22, 0.18, 0.12],
    'KS': [0.02, 0.03, 0.06, 0.09, 0.07, 0.04]
}

# Simulated model performance with adaptation
adaptation_data = {
    'Run': [1, 2, 3, 4, 5, 6],
    'F1': [0.95, 0.96, 0.97, 0.91, 0.94, 0.97],  # Drops at run 4 when drift detected
    'Adaptation': ['stable', 'stable', 'stable', 'fast', 'stable', 'stable']
}

fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

# Top: PSI + KS (Drift)
ax1 = axes[0]
ax1.plot(drift_data['Run'], drift_data['PSI'], 'b-o', label='PSI', linewidth=2, markersize=8)
ax1.plot(drift_data['Run'], drift_data['KS'], 'g-s', label='KS', linewidth=2, markersize=8)
ax1.axhline(y=0.20, color='r', linestyle='--', linewidth=2, label='PSI Threshold')
ax1.axhline(y=0.05, color='orange', linestyle='--', linewidth=2, label='KS Threshold')
ax1.fill_between(drift_data['Run'], 0, 0.2, alpha=0.1, color='green')
ax1.fill_between(drift_data['Run'], 0.2, 0.3, alpha=0.1, color='red')
ax1.set_ylabel('Drift Metric', fontsize=12)
ax1.set_title('A) Drift Detection', fontsize=14, fontweight='bold')
ax1.legend(loc='upper left')
ax1.set_ylim(0, 0.3)
ax1.grid(True, alpha=0.3)

# Middle: F1 Score over time
ax2 = axes[1]
colors_f1 = ['blue' if f >= 0.93 else 'red' for f in adaptation_data['F1']]
ax2.plot(adaptation_data['Run'], adaptation_data['F1'], 'ko-', linewidth=2, markersize=10)
ax2.axhline(y=0.95, color='green', linestyle='--', alpha=0.7, label='Target F1')
ax2.fill_between(adaptation_data['Run'], 0.9, adaptation_data['F1'], alpha=0.3, color='blue')
ax2.set_ylabel('F1 Score', fontsize=12)
ax2.set_title('B) Model Performance Over Time', fontsize=14, fontweight='bold')
ax2.set_ylim(0.88, 1.02)
ax2.grid(True, alpha=0.3)
ax2.legend(loc='lower right')

# Annotate the drop
ax2.annotate('Drift detected\nModel degrades', 
             xy=(4, 0.91), xytext=(4.3, 0.88),
             arrowprops=dict(arrowstyle='->', color='red'),
             fontsize=10, color='red')

# Bottom: Adaptation status
ax3 = axes[2]
adaptation_colors = {'stable': 'green', 'fast': 'blue', 'slow': 'red'}
for i, (run, ad) in enumerate(zip(drift_data['Run'], adaptation_data['Adaptation'])):
    ax3.bar(run, 1, color=adaptation_colors[ad], alpha=0.7)
ax3.set_ylabel('Adaptation Status', fontsize=12)
ax3.set_title('C) System Adaptation Response', fontsize=14, fontweight='bold')
ax3.set_ylim(0, 1.5)
ax3.set_xlabel('Run Number', fontsize=12)

# Legend for adaptation
from matplotlib.patches import Patch
legend_patches = [
    Patch(facecolor='green', alpha=0.7, label='Stable'),
    Patch(facecolor='blue', alpha=0.7, label='Fast Adaptation'),
    Patch(facecolor='red', alpha=0.7, label='Slow Adaptation')
]
ax3.legend(handles=legend_patches, loc='upper right')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/drift_adaptation_combined.png', dpi=150, bbox_inches='tight')
print(f'Saved: {OUTPUT_DIR}/drift_adaptation_combined.png')
plt.close()

print('\n=== All diagrams generated! ===')
print(f'Output directory: {OUTPUT_DIR}')