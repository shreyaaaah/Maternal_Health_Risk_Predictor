import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.decomposition import PCA

print("=" * 70)
print("STEP 2: EDA + PCA")
print("=" * 70)

X_raw = pd.read_csv("X_raw.csv")
X_scaled = pd.read_csv("X_scaled.csv")

# -----------------------------
# 1. BASIC INFO
# -----------------------------
print("\nDataset shape:", X_raw.shape)
print("\nMissing values:")
print(X_raw.isnull().sum())
print("\nSummary:")
print(X_raw.describe().round(2))

# -----------------------------
# 2. CORRELATION HEATMAP
# -----------------------------
plt.figure(figsize=(10, 7))
sns.heatmap(
    X_raw.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5
)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("eda_correlation_heatmap.png", dpi=150)
plt.close()

# -----------------------------
# 3. DISTRIBUTIONS
# -----------------------------
for col in X_raw.columns:
    plt.figure(figsize=(7, 4))
    sns.histplot(X_raw[col], kde=True)
    plt.title(f"Distribution of {col}")
    plt.tight_layout()
    plt.savefig(f"eda_dist_{col}.png", dpi=150)
    plt.close()

# -----------------------------
# 4. BOXPLOTS
# -----------------------------
plt.figure(figsize=(12, 6))
sns.boxplot(data=X_raw)
plt.xticks(rotation=30)
plt.title("Feature Boxplots")
plt.tight_layout()
plt.savefig("eda_boxplots.png", dpi=150)
plt.close()

# -----------------------------
# 5. PCA
# -----------------------------
pca_full = PCA()
pca_full.fit(X_scaled)

cum_var = np.cumsum(pca_full.explained_variance_ratio_)
n_components = np.argmax(cum_var >= 0.95) + 1

plt.figure(figsize=(8, 5))
plt.plot(range(1, len(cum_var) + 1), cum_var * 100, marker="o")
plt.axhline(95, linestyle="--")
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Variance Explained (%)")
plt.title("PCA Cumulative Variance")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("pca_variance.png", dpi=150)
plt.close()

pca = PCA(n_components=min(3, X_scaled.shape[1]), random_state=42)
X_pca = pca.fit_transform(X_scaled)

np.save("X_pca.npy", X_pca)
pickle.dump(pca, open("pca_model.pkl", "wb"))

print("=" * 70)
print("STEP 2 COMPLETE")
print(f"PCA components saved: {X_pca.shape[1]}")
print("Saved EDA plots + X_pca.npy + pca_model.pkl")
print("=" * 70)