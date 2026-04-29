import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.neighbors import NearestNeighbors

print("=" * 70)
print("STEP 3: CLUSTERING - KMEANS + DBSCAN + AGGLOMERATIVE")
print("=" * 70)

X_scaled = pd.read_csv("X_scaled.csv").values
X_pca = np.load("X_pca.npy")

# Use PCA space for clustering
X = X_pca

results = {}

# -----------------------------
# 1. KMEANS
# -----------------------------
kmeans = KMeans(n_clusters=3, random_state=42, n_init=20, max_iter=500)
labels_kmeans = kmeans.fit_predict(X)

results["KMeans"] = {
    "silhouette": silhouette_score(X, labels_kmeans),
    "davies_bouldin": davies_bouldin_score(X, labels_kmeans),
    "calinski": calinski_harabasz_score(X, labels_kmeans),
    "clusters": len(set(labels_kmeans))
}

pickle.dump(kmeans, open("kmeans_model.pkl", "wb"))
np.save("labels_kmeans.npy", labels_kmeans)

# -----------------------------
# 2. DBSCAN EPS TUNING
# -----------------------------
neighbors = NearestNeighbors(n_neighbors=10)
neighbors_fit = neighbors.fit(X)
distances, _ = neighbors_fit.kneighbors(X)
distances = np.sort(distances[:, -1])

plt.figure(figsize=(8, 5))
plt.plot(distances)
plt.title("DBSCAN k-distance Graph")
plt.xlabel("Points sorted by distance")
plt.ylabel("10th nearest neighbor distance")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("dbscan_k_distance.png", dpi=150)
plt.close()

best_dbscan = None
best_labels_dbscan = None
best_score = -1

for eps in np.arange(0.2, 2.5, 0.1):
    db = DBSCAN(eps=eps, min_samples=10)
    labels = db.fit_predict(X)

    non_noise = labels != -1
    n_clusters = len(set(labels[non_noise]))

    if n_clusters >= 2 and non_noise.sum() > 100:
        score = silhouette_score(X[non_noise], labels[non_noise])
        if score > best_score:
            best_score = score
            best_dbscan = db
            best_labels_dbscan = labels

if best_dbscan is None:
    best_dbscan = DBSCAN(eps=0.8, min_samples=10)
    best_labels_dbscan = best_dbscan.fit_predict(X)

pickle.dump(best_dbscan, open("dbscan_model.pkl", "wb"))
np.save("labels_dbscan.npy", best_labels_dbscan)

db_non_noise = best_labels_dbscan != -1
if len(set(best_labels_dbscan[db_non_noise])) > 1:
    db_sil = silhouette_score(X[db_non_noise], best_labels_dbscan[db_non_noise])
else:
    db_sil = -1

results["DBSCAN"] = {
    "silhouette": db_sil,
    "davies_bouldin": "N/A",
    "calinski": "N/A",
    "clusters": len(set(best_labels_dbscan)) - (1 if -1 in best_labels_dbscan else 0),
    "noise_points": int((best_labels_dbscan == -1).sum())
}

# -----------------------------
# 3. AGGLOMERATIVE
# -----------------------------
agg = AgglomerativeClustering(n_clusters=3, linkage="ward")
labels_agg = agg.fit_predict(X)

np.save("labels_agg.npy", labels_agg)

results["Agglomerative"] = {
    "silhouette": silhouette_score(X, labels_agg),
    "davies_bouldin": davies_bouldin_score(X, labels_agg),
    "calinski": calinski_harabasz_score(X, labels_agg),
    "clusters": len(set(labels_agg))
}

# -----------------------------
# 4. SAVE METRICS
# -----------------------------
metrics_df = pd.DataFrame(results).T
metrics_df.to_csv("clustering_metrics.csv")

print("\nClustering Metrics:")
print(metrics_df)

# -----------------------------
# 5. VISUALIZE CLUSTERS
# -----------------------------
def plot_clusters(labels, title, filename):
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=labels, s=8, alpha=0.6, cmap="viridis")
    plt.title(title)
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()

plot_clusters(labels_kmeans, "KMeans Clusters", "kmeans_clusters.png")
plot_clusters(best_labels_dbscan, "DBSCAN Clusters", "dbscan_clusters.png")
plot_clusters(labels_agg, "Agglomerative Clusters", "agglomerative_clusters.png")

print("=" * 70)
print("STEP 3 COMPLETE")
print("Saved clustering models, labels, metrics, and plots")
print("=" * 70)