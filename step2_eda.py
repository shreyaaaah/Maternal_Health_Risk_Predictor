"""
STEP 2: Exploratory Data Analysis + Dimensionality Reduction
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("  MATERNAL HEALTH CLUSTERING SYSTEM")
print("  Step 2: EDA + Dimensionality Reduction")
print("=" * 60)

# Load processed data
X_scaled = pd.read_csv('X_scaled.csv')
X_raw = pd.read_csv('X_raw.csv')
source = pd.read_csv('source_labels.csv')

print(f"\n[OK] Loaded: {X_scaled.shape[0]} rows, {X_scaled.shape[1]} features")

# ─────────────────────────────────────────────
# 1. CORRELATION HEATMAP
# ─────────────────────────────────────────────

print("\n[1/3] Generating correlation heatmap...")

fig, axes = plt.subplots(1, 2, figsize=(22, 9))

# Full correlation matrix
corr = X_raw.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, ax=axes[0], cmap='RdYlGn', center=0,
            annot=False, linewidths=0.3, vmin=-1, vmax=1,
            cbar_kws={'shrink': 0.8})
axes[0].set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold', pad=12)
axes[0].tick_params(axis='x', rotation=45, labelsize=7)
axes[0].tick_params(axis='y', labelsize=7)

# Distribution of key features by source
key_features = ['heart_rate', 'pct_abnormal_stv', 'mean_stv', 'histogram_mean', 'histogram_variance']
X_raw['source'] = source['source']
melted = X_raw[key_features + ['source']].melt(id_vars='source', var_name='feature', value_name='value')
sns.boxplot(data=melted, x='feature', y='value', hue='source', ax=axes[1],
            palette={'maternal': '#E76F51', 'ctg': '#2A9D8F'})
axes[1].set_title('Key Feature Distributions by Dataset Source', fontsize=14, fontweight='bold', pad=12)
axes[1].tick_params(axis='x', rotation=25, labelsize=8)
axes[1].legend(title='Source')

plt.suptitle('Maternal Health Combined Dataset — EDA', fontsize=16, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('eda_overview.png', dpi=150, bbox_inches='tight')
plt.close()
print("      [OK] Saved: eda_overview.png")

# ─────────────────────────────────────────────
# 2. PCA — Variance Explained
# ─────────────────────────────────────────────

print("\n[2/3] Running PCA...")

pca = PCA()
pca.fit(X_scaled)

cumvar = np.cumsum(pca.explained_variance_ratio_)
n_components_95 = np.argmax(cumvar >= 0.95) + 1

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Scree plot
axes[0].bar(range(1, 21),
            pca.explained_variance_ratio_[:20] * 100,
            color='#2A9D8F', alpha=0.8, edgecolor='white')
axes[0].set_xlabel('Principal Component')
axes[0].set_ylabel('Variance Explained (%)')
axes[0].set_title('PCA Scree Plot', fontweight='bold')
axes[0].axhline(y=5, color='red', linestyle='--', alpha=0.5, label='5% threshold')

# Cumulative variance
axes[1].plot(range(1, len(cumvar)+1), cumvar * 100, 'o-', color='#E76F51', linewidth=2)
axes[1].axhline(y=95, color='gray', linestyle='--', alpha=0.7, label='95% variance')
axes[1].axvline(x=n_components_95, color='#2A9D8F', linestyle='--', alpha=0.7,
                label=f'{n_components_95} components')
axes[1].set_xlabel('Number of Components')
axes[1].set_ylabel('Cumulative Variance Explained (%)')
axes[1].set_title('Cumulative Variance Explained', fontweight='bold')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.suptitle(f'PCA Analysis — {n_components_95} components explain 95% variance',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('pca_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"      [OK] {n_components_95} components explain 95% variance")
print("      [OK] Saved: pca_analysis.png")

# PCA transform for clustering (preserving more components for nuance)
n_comp = min(8, X_scaled.shape[1])
pca_final = PCA(n_components=n_comp)
X_pca = pca_final.fit_transform(X_scaled)

import pickle
pickle.dump(pca_final, open('pca_model.pkl', 'wb'))
np.save('X_pca.npy', X_pca)

# ─────────────────────────────────────────────
# 3. UMAP — Visual Exploration
# ─────────────────────────────────────────────

print("\n[3/3] Running UMAP (this takes ~1-2 mins)...")

reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, n_components=2, random_state=42)
X_umap = reducer.fit_transform(X_scaled)

np.save('X_umap.npy', X_umap)
pickle.dump(reducer, open('umap_model.pkl', 'wb'))

fig, ax = plt.subplots(figsize=(10, 7))
colors = source['source'].map({'maternal': '#E76F51', 'ctg': '#2A9D8F'})
scatter = ax.scatter(X_umap[:, 0], X_umap[:, 1], c=colors, alpha=0.5, s=8)
ax.set_title('UMAP Projection — Combined Dataset\n(Color = Data Source)',
             fontsize=14, fontweight='bold')
ax.set_xlabel('UMAP Dimension 1')
ax.set_ylabel('UMAP Dimension 2')

from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#E76F51', label=f"Maternal (n={source['source'].value_counts()['maternal']})"),
                   Patch(facecolor='#2A9D8F', label=f"CTG (n={source['source'].value_counts()['ctg']})")]
ax.legend(handles=legend_elements, loc='upper right')
ax.grid(alpha=0.2)

plt.tight_layout()
plt.savefig('umap_preview.png', dpi=150, bbox_inches='tight')
plt.close()
print("      [OK] Saved: umap_preview.png")

print("\n" + "=" * 60)
print("  [SUCCESS] Step 2 Complete!")
print(f"  PCA reduced dimensions: {X_scaled.shape[1]} -> {n_components_95}")
print("  Saved: X_pca.npy, X_umap.npy, umap_model.pkl, pca_model.pkl")
print("=" * 60)
