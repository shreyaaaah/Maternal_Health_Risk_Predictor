# 🏥 Maternal Health Risk Clustering — Real-Time System
### Unsupervised Learning Project | 20-Day Plan

---

## 📌 Project Summary
A real-time unsupervised clustering system that discovers hidden maternal
health risk profiles by combining two UCI datasets:
- UCI Maternal Health Risk (1,014 records, 6 features)
- UCI CTG Cardiotocography (2,126 records, 21 features)
**Total: ~3,140 patients, 27 features — zero labels used during training**

---

## 🗂️ Project Structure

```
maternal_health_clustering/
│
├── step1_data_fusion.py      # Load + merge + preprocess both datasets
├── step2_eda.py              # EDA, PCA, UMAP visualization
├── step3_clustering.py       # K-Means + HDBSCAN + Agglomerative
├── step4_profiling.py        # Cluster naming & interpretation
├── step5_dashboard.py        # Real-time Streamlit dashboard
├── requirements.txt          # All dependencies
└── README.md                 # This file
```

---

## ⚙️ Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run all steps in order
python step1_data_fusion.py
python step2_eda.py
python step3_clustering.py
python step4_profiling.py

# 3. Launch real-time dashboard
streamlit run step5_dashboard.py
```

---

## 📅 20-Day Timeline

| Days  | Step  | Task                                      |
|-------|-------|-------------------------------------------|
| 1–2   | Setup | Install deps, understand datasets         |
| 3–5   | Step 1| Data fusion, feature alignment, scaling   |
| 6–7   | Step 2| EDA, correlation heatmap, PCA, UMAP       |
| 8–13  | Step 3| Clustering — tune K-Means, HDBSCAN, Agg  |
| 14–15 | Step 4| Profile & name clusters                   |
| 16–17 | Step 5| Build real-time prediction engine         |
| 18–19 | Step 5| Build Streamlit dashboard                 |
| 20    | Final | Test, polish, write report                |

---

## 🔬 Algorithms Used

| Algorithm        | Purpose                              |
|------------------|--------------------------------------|
| K-Means          | Primary clustering (fast, tunable)   |
| HDBSCAN          | Density-based (finds irregular shapes)|
| Agglomerative    | Hierarchical clustering (comparison) |
| PCA              | Dimensionality reduction pre-cluster |
| UMAP             | 2D visualization of clusters         |
| SimpleImputer    | Handle missing values across datasets|
| StandardScaler   | Normalize all features               |

---

## 📊 Evaluation Metrics
- **Silhouette Score** — measures cluster compactness & separation
- **Davies-Bouldin Index** — lower = better separated clusters
- **Calinski-Harabasz Score** — higher = denser, well-separated clusters
- **ARI (optional)** — compare discovered clusters vs dropped labels

---

## ⚠️ Important Notes
1. The `RiskLevel` (Maternal) and `NSP` (CTG) columns are DROPPED before
   any clustering — this is a purely unsupervised system.
2. Labels are only used optionally at the end for validation (ARI score).
3. This is an academic/research project — not a clinical decision tool.

---

## 📦 Datasets
- Maternal Health Risk: https://archive.ics.uci.edu/dataset/863/maternal+health+risk
- Cardiotocography: https://archive.ics.uci.edu/dataset/193/cardiotocography
