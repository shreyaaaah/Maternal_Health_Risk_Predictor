# 🏥 Maternal Health Risk Clustering — Real-Time System
### Unsupervised Learning Project

---

## 📌 Project Summary
A real-time unsupervised clustering system that discovers hidden maternal health risk profiles. The final model is trained exclusively on 6 core maternal physiological features (Age, Systolic BP, Diastolic BP, Blood Sugar, Body Temperature, Heart Rate).

To create a robust 10,000-patient dataset, we combined:
- **UCI Maternal Health Risk Dataset** (1,014 real records)
- **UCI CTG Cardiotocography Dataset** (2,126 records) — *Note: Raw fetal CTG features were not used in the model. Instead, fetal stress indicators were mathematically transformed into proxy maternal physiological features to augment the dataset.*

**Total after Data Fusion and Augmentation: 10,000 patients — zero labels used during training**

---

## 🗂️ Project Structure

```
maternal_health_clustering/
│
├── step1_data_fusion.py          # Load, transform CTG to maternal proxies, augment to 10K
├── step2_eda.py                  # Exploratory Data Analysis & PCA
├── step3_clustering.py           # K-Means, DBSCAN, and Agglomerative Clustering
├── step4_cluster_profiling.py    # Cluster interpretation & clinical risk scoring
├── step5_dashboard.py            # Real-time Streamlit dashboard
├── requirements.txt              # All dependencies
└── README.md                     # This file
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
python step4_cluster_profiling.py

# 3. Launch real-time dashboard
streamlit run step5_dashboard.py
```

---

## 🔬 Algorithms Used

| Algorithm        | Purpose                              |
|------------------|--------------------------------------|
| K-Means          | Primary clustering (fast, tunable)   |
| DBSCAN           | Density-based (finds irregular shapes)|
| Agglomerative    | Hierarchical clustering (comparison) |
| PCA              | Dimensionality reduction & 2D Viz    |
| SimpleImputer    | Handle missing values                |
| StandardScaler   | Normalize all features               |

---

## 📊 Evaluation Metrics
- **Silhouette Score** — measures cluster compactness & separation
- **Davies-Bouldin Index** — lower = better separated clusters
- **Calinski-Harabasz Score** — higher = denser, well-separated clusters

---

## ⚠️ Important Notes
1. The original clinical labels (`RiskLevel` for Maternal, `NSP` for CTG) are DROPPED before any clustering — this is a purely unsupervised system.
2. Clinical thresholds are used only in the final profiling step to give medical interpretability to the discovered clusters.
3. This is an academic/research project — not a clinical decision tool.

---

## 📦 Datasets
- Maternal Health Risk: https://archive.ics.uci.edu/dataset/863/maternal+health+risk
- Cardiotocography: https://archive.ics.uci.edu/dataset/193/cardiotocography
