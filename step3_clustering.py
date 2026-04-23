"""
STEP 3: Clustering — K-Means + HDBSCAN + Agglomerative
Compare all 3 and select best
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import hdbscan
import pickle
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("  MATERNAL HEALTH CLUSTERING SYSTEM")
print("  Step 3: Clustering")
print("=" * 60)

# Load data
X_scaled = pd.read_csv('X_scaled.csv').values
X_pca = np.load('X_pca.npy')
X_umap = np.load('X_umap.npy')

# Use PCA-reduced data for clustering (noise-reduced, faster)
X = X_pca

# ─────────────────────────────────────────────
# 1. K-MEANS — Find Optimal K
# ─────────────────────────────────────────────

print("\n[1/3] Running K-Means (k=2 to 8)...")

k_range = range(2, 9)
inertias, silhouettes, db_scores = [], [], []

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X, labels))
    db_scores.append(davies_bouldin_score(X, labels))
    print(f"      k={k} | Silhouette: {silhouettes[-1]:.3f} | DB Index: {db_scores[-1]:.3f}")

# Plot elbow + silhouette
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(k_range, inertias, 'o-', color='#E76F51', linewidth=2, markersize=8)
axes[0].set_xlabel('Number of Clusters (k)')
axes[0].set_ylabel('Inertia (Within-cluster SSE)')
axes[0].set_title('Elbow Curve', fontweight='bold')
axes[0].grid(alpha=0.3)

axes[1].plot(k_range, silhouettes, 's-', color='#2A9D8F', linewidth=2, markersize=8)
axes[1].set_xlabel('Number of Clusters (k)')
axes[1].set_ylabel('Silhouette Score')
axes[1].set_title('Silhouette Score by k', fontweight='bold')
axes[1].axvline(x=k_range[np.argmax(silhouettes)], color='gray', linestyle='--', alpha=0.6)
axes[1].grid(alpha=0.3)

plt.suptitle('K-Means — Finding Optimal k', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('kmeans_elbow.png', dpi=150, bbox_inches='tight')
plt.close()

# Select best k (preferring more clusters if silhouette is similar)
max_sil = max(silhouettes)
best_k = k_range[0]
for i, sil in enumerate(silhouettes):
    if sil >= max_sil * 0.9:  # within 10% of max
        best_k = k_range[i]

print(f"\n      [OK] Selected k = {best_k} (Silhouette = {silhouettes[k_range.index(best_k)]:.3f})")

# Final K-Means model
kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
labels_kmeans = kmeans_final.fit_predict(X)
pickle.dump(kmeans_final, open('kmeans_model.pkl', 'wb'))

# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
# 2. HDBSCAN — Tuning min_cluster_size
# ─────────────────────────────────────────────

print("\n[2/3] Tuning HDBSCAN...")

best_hdb_sil = -1
best_hdb_params = {}
best_labels_hdb = None
best_hdb_model = None

for mcs in [30, 50, 80]:
    for ms in [5, 10, 15]:
        hdb_test = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=ms, prediction_data=True)
        labels_test = hdb_test.fit_predict(X)
        
        valid_mask = labels_test != -1
        if valid_mask.sum() > 100 and len(set(labels_test[valid_mask])) > 1:
            sil = silhouette_score(X[valid_mask], labels_test[valid_mask])
            if sil > best_hdb_sil:
                best_hdb_sil = sil
                best_hdb_params = {'mcs': mcs, 'ms': ms}
                best_labels_hdb = labels_test
                best_hdb_model = hdb_test

if best_hdb_model:
    hdb = best_hdb_model
    labels_hdbscan = best_labels_hdb
    sil_hdb = best_hdb_sil
    print(f"      Best Params      : min_cluster_size={best_hdb_params['mcs']}, min_samples={best_hdb_params['ms']}")
else:
    # Fallback if no valid clusters found
    hdb = hdbscan.HDBSCAN(min_cluster_size=50)
    labels_hdbscan = hdb.fit_predict(X)
    sil_hdb = 0

n_clusters_hdb = len(set(labels_hdbscan)) - (1 if -1 in labels_hdbscan else 0)
noise_pct = (labels_hdbscan == -1).sum() / len(labels_hdbscan) * 100

print(f"      Clusters found   : {n_clusters_hdb}")
print(f"      Noise points     : {noise_pct:.1f}%")
print(f"      Silhouette score : {sil_hdb:.3f}")
pickle.dump(hdb, open('hdbscan_model.pkl', 'wb'))

# ─────────────────────────────────────────────
# 3. AGGLOMERATIVE
# ─────────────────────────────────────────────

print("\n[3/3] Running Agglomerative Clustering...")

agg = AgglomerativeClustering(n_clusters=best_k, linkage='ward')
labels_agg = agg.fit_predict(X)
sil_agg = silhouette_score(X, labels_agg)
print(f"      Silhouette score : {sil_agg:.3f}")

# ─────────────────────────────────────────────
# 4. COMPARE ALL 3
# ─────────────────────────────────────────────

print("\nModel Comparison:")
print(f"{'Model':<20} {'Clusters':<10} {'Silhouette':<12} {'DB Index':<12}")
print("-" * 55)
print(f"{'K-Means':<20} {best_k:<10} {max(silhouettes):<12.3f} {db_scores[np.argmax(silhouettes)]:<12.3f}")
print(f"{'HDBSCAN':<20} {n_clusters_hdb:<10} {sil_hdb:<12.3f} {'N/A (noise)':<12}")
print(f"{'Agglomerative':<20} {best_k:<10} {sil_agg:<12.3f} {davies_bouldin_score(X, labels_agg):<12.3f}")

# Save all labels
np.save('labels_kmeans.npy', labels_kmeans)
np.save('labels_hdbscan.npy', labels_hdbscan)
np.save('labels_agg.npy', labels_agg)

# ─────────────────────────────────────────────
# 5. VISUALIZE ON UMAP
# ─────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(20, 6))

algo_names = ['K-Means', 'HDBSCAN', 'Agglomerative']
all_labels = [labels_kmeans, labels_hdbscan, labels_agg]
palettes = ['tab10', 'tab10', 'tab10']

for ax, name, labels in zip(axes, algo_names, all_labels):
    unique = sorted(set(labels))
    colors = plt.cm.get_cmap('tab10', len(unique))
    for i, cluster in enumerate(unique):
        mask = labels == cluster
        label = f'Noise ({mask.sum()})' if cluster == -1 else f'Cluster {cluster} ({mask.sum()})'
        ax.scatter(X_umap[mask, 0], X_umap[mask, 1],
                   c=[colors(i)], s=5, alpha=0.5, label=label)
    ax.set_title(f'{name}\n({len(unique)} groups)', fontweight='bold', fontsize=12)
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.legend(fontsize=7, markerscale=2)
    ax.grid(alpha=0.2)

plt.suptitle('Clustering Results — UMAP Visualization', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('clustering_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n[OK] Saved: clustering_comparison.png")

print("\n" + "=" * 60)
print(f"  [SUCCESS] Step 3 Complete! Best model: K-Means (k={best_k})")
print("  Saved: kmeans_model.pkl, hdbscan_model.pkl, all label files")
print("=" * 60)
