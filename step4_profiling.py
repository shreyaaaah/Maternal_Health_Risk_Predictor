"""
STEP 4: Cluster Profiling — Name & Interpret Each Cluster
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("  MATERNAL HEALTH CLUSTERING SYSTEM")
print("  Step 4: Cluster Profiling")
print("=" * 60)

# Load everything
X_raw = pd.read_csv('X_raw.csv')
labels = np.load('labels_kmeans.npy')
source = pd.read_csv('source_labels.csv')

X_raw['cluster'] = labels
X_raw['source'] = source['source']

n_clusters = len(set(labels))
print(f"\n[OK] Loaded {len(X_raw)} patients across {n_clusters} clusters")

# ─────────────────────────────────────────────
# 1. CLUSTER PROFILES — Mean Stats
# ─────────────────────────────────────────────

print("\n[1/3] Computing cluster profiles...")

profile = X_raw.groupby('cluster').agg({
    'age': 'mean',
    'systolic_bp': 'mean',
    'diastolic_bp': 'mean',
    'blood_sugar': 'mean',
    'body_temp': 'mean',
    'heart_rate': 'mean',
    'pct_abnormal_stv': 'mean',
    'mean_stv': 'mean',
    'uterine_contractions': 'mean',
    'histogram_variance': 'mean',
    'cluster': 'count'
}).rename(columns={'cluster': 'count'}).round(2)

print("\nCluster Mean Profiles:")
print(profile.to_string())

# ─────────────────────────────────────────────
# 2. AUTO-NAME CLUSTERS BASED ON VITALS
# ─────────────────────────────────────────────

print("\n[2/3] Auto-naming clusters...")

def name_cluster(row, all_profiles):
    """Name cluster based on relative vitals compared to overall mean"""
    overall = all_profiles.mean()
    
    risk_score = 0
    flags = []
    
    # Blood pressure check (Preeclampsia risk)
    if row.get('systolic_bp', 0) > overall.get('systolic_bp', 0) * 1.05:
        risk_score += 3
        flags.append("Hypertension (Preeclampsia Risk)")
    
    # Blood sugar check (Gestational Diabetes risk)
    if row.get('blood_sugar', 0) > overall.get('blood_sugar', 0) * 1.1:
        risk_score += 3
        flags.append("Hyperglycemia (Gestational Diabetes Risk)")
    
    # Heart rate
    if row.get('heart_rate', 0) > overall.get('heart_rate', 0) * 1.05:
        risk_score += 1
        flags.append("Maternal Tachycardia")
    
    # Fetal variability (CTG) - Critical
    if row.get('pct_abnormal_stv', 0) > overall.get('pct_abnormal_stv', 0) * 1.1:
        risk_score += 4
        flags.append("Abnormal Fetal HRV (Hypoxia Risk)")
    
    # Histogram variance
    if row.get('histogram_variance', 0) > overall.get('histogram_variance', 0) * 1.2:
        risk_score += 2
        flags.append("Fetal Pattern Irregularity")

    if risk_score >= 5:
        return "HIGH RISK (Critical)", flags
    elif risk_score >= 2:
        return "MEDIUM RISK (Monitor)", flags
    else:
        return "LOW RISK (Stable)", flags

cluster_names = {}
for cluster_id in range(n_clusters):
    row = profile.loc[cluster_id] if cluster_id in profile.index else profile.iloc[0]
    name, flags = name_cluster(row, profile)
    cluster_names[cluster_id] = name
    count = int(profile.loc[cluster_id, 'count']) if cluster_id in profile.index else 0
    print(f"\n  Cluster {cluster_id}: {name}")
    print(f"  Patients : {count}")
    print(f"  Key flags: {', '.join(flags) if flags else 'Normal vitals'}")

# Save cluster names
import json
json.dump(cluster_names, open('cluster_names.json', 'w'))

# ─────────────────────────────────────────────
# 3. VISUALIZE CLUSTER PROFILES
# ─────────────────────────────────────────────

print("\n[3/3] Generating cluster profile charts...")

key_features = ['heart_rate', 'systolic_bp', 'diastolic_bp',
                'blood_sugar', 'body_temp', 'pct_abnormal_stv',
                'mean_stv', 'histogram_variance']

# Radar / parallel coordinates style — heatmap of z-scores
profile_z = profile[key_features].copy()
for col in key_features:
    mean, std = profile_z[col].mean(), profile_z[col].std()
    profile_z[col] = (profile_z[col] - mean) / (std + 1e-8)

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# Heatmap of z-scores
labels_display = [f"Cluster {i}\n{cluster_names[i]}" for i in profile_z.index if i in cluster_names]
sns.heatmap(profile_z, ax=axes[0], cmap='RdYlGn_r', center=0, annot=True, fmt='.2f',
            yticklabels=labels_display, linewidths=0.5, cbar_kws={'label': 'Z-Score'})
axes[0].set_title('Cluster Profiles — Z-Score Heatmap\n(Red = High Risk, Green = Low Risk)',
                  fontweight='bold', fontsize=12)
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=30, ha='right', fontsize=9)

# Cluster size pie
sizes = [profile.loc[i, 'count'] for i in profile.index]
cluster_labels_pie = [f"Cluster {i}\n{cluster_names.get(i,'')}\n({int(sizes[j])} pts)"
                      for j, i in enumerate(profile.index)]
colors_pie = ['#2ECC71', '#F39C12', '#E74C3C', '#3498DB', '#9B59B6'][:n_clusters]
axes[1].pie(sizes, labels=cluster_labels_pie, colors=colors_pie,
            autopct='%1.1f%%', startangle=90,
            textprops={'fontsize': 9}, pctdistance=0.75)
axes[1].set_title('Cluster Size Distribution', fontweight='bold', fontsize=12)

plt.suptitle('Maternal Health — Discovered Risk Clusters', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('cluster_profiles.png', dpi=150, bbox_inches='tight')
plt.close()
print("      [OK] Saved: cluster_profiles.png")

# Box plots for key features
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()

cluster_colors = ['#2ECC71', '#F39C12', '#E74C3C', '#3498DB', '#9B59B6']

for i, feat in enumerate(key_features):
    data_plot = [X_raw[X_raw['cluster'] == c][feat].dropna().values for c in range(n_clusters)]
    bp = axes[i].boxplot(data_plot, patch_artist=True, notch=False)
    for patch, color in zip(bp['boxes'], cluster_colors[:n_clusters]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[i].set_title(feat.replace('_', ' ').title(), fontweight='bold', fontsize=10)
    axes[i].set_xticklabels([f'C{j}' for j in range(n_clusters)], fontsize=8)
    axes[i].grid(alpha=0.3, axis='y')

plt.suptitle('Feature Distribution per Cluster', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('cluster_boxplots.png', dpi=150, bbox_inches='tight')
plt.close()
print("      [OK] Saved: cluster_boxplots.png")

print("\n" + "=" * 60)
print("  [SUCCESS] Step 4 Complete! Clusters profiled and named.")
print("  Saved: cluster_profiles.png, cluster_boxplots.png")
print("=" * 60)
