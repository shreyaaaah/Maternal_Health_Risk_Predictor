import pandas as pd
import numpy as np
import pickle
import os
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

np.random.seed(42)

print("=" * 70)
print("STEP 1: DATA FUSION + 10K AUGMENTATION")
print("=" * 70)

FEATURES = [
    "age",
    "systolic_bp",
    "diastolic_bp",
    "blood_sugar",
    "body_temp",
    "heart_rate"
]

# -----------------------------
# 1. LOAD MATERNAL DATASET
# -----------------------------
if os.path.exists("Maternal Health Risk Data Set.csv"):
    df_maternal = pd.read_csv("Maternal Health Risk Data Set.csv")
else:
    maternal = fetch_ucirepo(id=863)
    df_maternal = maternal.data.features.copy()

df_maternal = df_maternal.rename(columns={
    "Age": "age",
    "SystolicBP": "systolic_bp",
    "DiastolicBP": "diastolic_bp",
    "BS": "blood_sugar",
    "BodyTemp": "body_temp",
    "HeartRate": "heart_rate"
})

df_maternal = df_maternal[FEATURES].copy()
df_maternal["source"] = "maternal_real"

print(f"Maternal dataset loaded: {df_maternal.shape}")

# -----------------------------
# 2. LOAD CTG DATASET
# -----------------------------
ctg = fetch_ucirepo(id=193)
df_ctg = ctg.data.features.copy()

df_ctg = df_ctg.rename(columns={
    "LB": "fhr_baseline",
    "AC": "accelerations",
    "FM": "fetal_movement",
    "UC": "uterine_contractions",
    "ASTV": "abnormal_stv",
    "ALTV": "abnormal_ltv"
})

print(f"CTG dataset loaded: {df_ctg.shape}")

# -----------------------------
# 3. TRANSFORM CTG INTO MATERNAL-LIKE FEATURE SPACE
# -----------------------------
n = len(df_ctg)

df_ctg_transformed = pd.DataFrame()

# Age generated realistically
df_ctg_transformed["age"] = np.random.normal(28, 6, n).round()
df_ctg_transformed["age"] = df_ctg_transformed["age"].clip(18, 45)

# CTG stress indicators
uc = df_ctg["uterine_contractions"].fillna(df_ctg["uterine_contractions"].median())
astv = df_ctg["abnormal_stv"].fillna(df_ctg["abnormal_stv"].median())
altv = df_ctg["abnormal_ltv"].fillna(df_ctg["abnormal_ltv"].median())
fhr = df_ctg["fhr_baseline"].fillna(df_ctg["fhr_baseline"].median())

# Convert CTG stress into maternal-equivalent physiological proxies
stress_score = (
    (uc - uc.min()) / (uc.max() - uc.min() + 1e-8) * 0.4 +
    (astv - astv.min()) / (astv.max() - astv.min() + 1e-8) * 0.4 +
    (altv - altv.min()) / (altv.max() - altv.min() + 1e-8) * 0.2
)

df_ctg_transformed["systolic_bp"] = 105 + stress_score * 45 + np.random.normal(0, 6, n)
df_ctg_transformed["diastolic_bp"] = 65 + stress_score * 25 + np.random.normal(0, 4, n)
df_ctg_transformed["blood_sugar"] = 5.8 + stress_score * 4 + np.random.normal(0, 0.4, n)
df_ctg_transformed["body_temp"] = 98.2 + stress_score * 3 + np.random.normal(0, 0.3, n)

# DO NOT directly use fetal HR as maternal HR.
# Convert fetal stress into maternal HR proxy.
df_ctg_transformed["heart_rate"] = 72 + stress_score * 30 + np.random.normal(0, 7, n)

df_ctg_transformed["source"] = "ctg_transformed"

# Clinical clipping
df_ctg_transformed["age"] = df_ctg_transformed["age"].clip(10, 60)
df_ctg_transformed["systolic_bp"] = df_ctg_transformed["systolic_bp"].clip(70, 180)
df_ctg_transformed["diastolic_bp"] = df_ctg_transformed["diastolic_bp"].clip(45, 120)
df_ctg_transformed["blood_sugar"] = df_ctg_transformed["blood_sugar"].clip(4, 20)
df_ctg_transformed["body_temp"] = df_ctg_transformed["body_temp"].clip(95, 105)
df_ctg_transformed["heart_rate"] = df_ctg_transformed["heart_rate"].clip(45, 150)

print(f"CTG transformed dataset: {df_ctg_transformed.shape}")

# -----------------------------
# 4. COMBINE
# -----------------------------
df_combined = pd.concat([df_maternal, df_ctg_transformed], ignore_index=True)

print(f"Combined before augmentation: {df_combined.shape}")

# -----------------------------
# 5. AUGMENT TO 10K ROWS
# -----------------------------
df_aug = df_combined.sample(n=10000, replace=True, random_state=42).reset_index(drop=True)

noise_config = {
    "age": 1.5,
    "systolic_bp": 3.5,
    "diastolic_bp": 2.5,
    "blood_sugar": 0.25,
    "body_temp": 0.15,
    "heart_rate": 3.0
}

for col, noise in noise_config.items():
    df_aug[col] = df_aug[col] + np.random.normal(0, noise, len(df_aug))

df_aug["age"] = df_aug["age"].round().clip(10, 60)
df_aug["systolic_bp"] = df_aug["systolic_bp"].clip(70, 200)
df_aug["diastolic_bp"] = df_aug["diastolic_bp"].clip(40, 130)
df_aug["blood_sugar"] = df_aug["blood_sugar"].clip(4, 20)
df_aug["body_temp"] = df_aug["body_temp"].clip(95, 105)
df_aug["heart_rate"] = df_aug["heart_rate"].clip(45, 160)

df_aug["source"] = "augmented_10k"

# -----------------------------
# 6. FINAL DATA
# -----------------------------
X_raw = df_aug[FEATURES].copy()

imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X_raw)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

X_raw.to_csv("X_raw.csv", index=False)
pd.DataFrame(X_scaled, columns=FEATURES).to_csv("X_scaled.csv", index=False)
df_aug[["source"]].to_csv("source_labels.csv", index=False)

pickle.dump(imputer, open("imputer.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
pickle.dump(FEATURES, open("all_features.pkl", "wb"))

print("=" * 70)
print("STEP 1 COMPLETE")
print("Saved:")
print("X_raw.csv")
print("X_scaled.csv")
print("source_labels.csv")
print("imputer.pkl")
print("scaler.pkl")
print("all_features.pkl")
print("=" * 70)
print(X_raw.describe().round(2))