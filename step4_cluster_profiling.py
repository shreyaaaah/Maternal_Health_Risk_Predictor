import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 70)
print("STEP 4: CLUSTER PROFILING")
print("=" * 70)

X_raw = pd.read_csv("X_raw.csv")
labels = np.load("labels_kmeans.npy")

X_raw["cluster"] = labels

# -----------------------------
# 1. CLUSTER PROFILE
# -----------------------------
profile = X_raw.groupby("cluster").agg({
    "age": "mean",
    "systolic_bp": "mean",
    "diastolic_bp": "mean",
    "blood_sugar": "mean",
    "body_temp": "mean",
    "heart_rate": "mean",
    "cluster": "count"
}).rename(columns={"cluster": "count"}).round(2)

print("\nCluster Profiles:")
print(profile)

# -----------------------------
# 2. RISK SCORING FUNCTION
# -----------------------------
def risk_score(row):
    score = 0
    flags = []

    if row["systolic_bp"] >= 140:
        score += 4
        flags.append("High systolic BP")
    elif row["systolic_bp"] >= 130:
        score += 2
        flags.append("Elevated systolic BP")
    elif row["systolic_bp"] >= 120:
        score += 1
        flags.append("Borderline systolic BP")

    if row["diastolic_bp"] >= 90:
        score += 3
        flags.append("High diastolic BP")
    elif row["diastolic_bp"] >= 80:
        score += 1
        flags.append("Borderline diastolic BP")

    if row["blood_sugar"] >= 8.0:
        score += 3
        flags.append("High blood sugar")
    elif row["blood_sugar"] >= 7.0:
        score += 1
        flags.append("Borderline blood sugar")

    if row["body_temp"] >= 100.4:
        score += 3
        flags.append("Fever")
    elif row["body_temp"] >= 99.5:
        score += 1
        flags.append("Slightly high temperature")

    if row["heart_rate"] >= 110:
        score += 3
        flags.append("High heart rate")
    elif row["heart_rate"] >= 100:
        score += 1
        flags.append("Borderline high heart rate")

    if row["age"] < 18 or row["age"] > 40:
        score += 1
        flags.append("Age-related risk")

    return score, flags

# -----------------------------
# 3. ASSIGN CLUSTER NAMES
# -----------------------------
cluster_names = {}
cluster_flags = {}

for cid, row in profile.iterrows():
    score, flags = risk_score(row)

    if score >= 5:
        name = "HIGH RISK"
    elif score >= 2:
        name = "MEDIUM RISK"
    else:
        name = "LOW RISK"

    cluster_names[str(cid)] = name
    cluster_flags[str(cid)] = flags

# If all clusters got same label, rank by risk score
if len(set(cluster_names.values())) == 1:
    print("\nAll clusters got same risk label. Applying rank-based assignment.")

    scores = {}
    for cid, row in profile.iterrows():
        score, _ = risk_score(row)
        scores[cid] = score

    ranked = sorted(scores, key=scores.get)

    cluster_names[str(ranked[0])] = "LOW RISK"
    cluster_names[str(ranked[1])] = "MEDIUM RISK"
    cluster_names[str(ranked[2])] = "HIGH RISK"

json.dump(cluster_names, open("cluster_names.json", "w"), indent=4)
json.dump(cluster_flags, open("cluster_flags.json", "w"), indent=4)

print("\nFinal Cluster Mapping:")
for cid, name in cluster_names.items():
    print(f"Cluster {cid} -> {name}")

# -----------------------------
# 4. SAVE PROFILE
# -----------------------------
profile.to_csv("cluster_profile.csv")

# -----------------------------
# 5. VISUALIZATION
# -----------------------------
plt.figure(figsize=(10, 6))
sns.heatmap(profile.drop(columns=["count"]), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Cluster Profile Heatmap")
plt.tight_layout()
plt.savefig("cluster_profile_heatmap.png", dpi=150)
plt.close()

plt.figure(figsize=(8, 5))
X_raw["risk_name"] = X_raw["cluster"].astype(str).map(cluster_names)
sns.countplot(data=X_raw, x="risk_name", order=["LOW RISK", "MEDIUM RISK", "HIGH RISK"])
plt.title("Cluster Risk Distribution")
plt.tight_layout()
plt.savefig("cluster_risk_distribution.png", dpi=150)
plt.close()

X_raw.to_csv("X_raw_labeled.csv", index=False)

print("=" * 70)
print("STEP 4 COMPLETE")
print("Saved cluster_names.json, cluster_profile.csv, plots")
print("=" * 70)