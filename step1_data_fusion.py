"""
STEP 1: Data Collection & Fusion
Maternal Health Risk (UCI) + CTG Cardiotocography (UCI)
"""

import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("  MATERNAL HEALTH CLUSTERING SYSTEM")
print("  Step 1: Data Collection & Fusion")
print("=" * 60)

import os
import time

# ─────────────────────────────────────────────
# 1. LOAD DATASETS
# ─────────────────────────────────────────────

print("\n[1/5] Loading datasets...")

# Load Maternal dataset (Local is preferred as it's the full 1014 records)
if os.path.exists("Maternal Health Risk Data Set.csv"):
    df_maternal = pd.read_csv("Maternal Health Risk Data Set.csv")
    print(f"      [OK] Loaded Maternal dataset from local CSV")
else:
    print("      Fetching Maternal dataset from UCI...")
    maternal = fetch_ucirepo(id=863)
    df_maternal = maternal.data.features.copy()

df_maternal['source'] = 'maternal'

# Load CTG dataset (Requires UCI fetch as local CTG.csv is just metadata)
df_ctg = None
for attempt in range(3):
    try:
        print(f"      Fetching CTG dataset from UCI (Attempt {attempt+1})...")
        ctg = fetch_ucirepo(id=193)
        df_ctg = ctg.data.features.copy()
        print(f"      [OK] Loaded: {df_ctg.shape[0]} rows")
        break
    except Exception as e:
        print(f"      [!] Fetch failed: {e}")
        time.sleep(2)

if df_ctg is None:
    print("      [!] Could not fetch CTG. Using Maternal data only for this run.")
    df_ctg = pd.DataFrame(columns=df_maternal.columns)

df_ctg['source'] = 'ctg'

# ─────────────────────────────────────────────
# 2. FEATURE ALIGNMENT STRATEGY
# ─────────────────────────────────────────────

print("\n[3/5] Aligning features across datasets...")

# Maternal features: Age, SystolicBP, DiastolicBP, BS (Blood Sugar), BodyTemp, HeartRate
# CTG features: LB (baseline FHR), AC, FM, UC, ASTV, MSTV, ALTV, MLTV, DL, DS, DP,
#               Width, Min, Max, Nmax, Nzeros, Mode, Mean, Median, Variance, Tendency

# Rename maternal columns for clarity
df_maternal = df_maternal.rename(columns={
    'Age': 'age',
    'SystolicBP': 'systolic_bp',
    'DiastolicBP': 'diastolic_bp',
    'BS': 'blood_sugar',
    'BodyTemp': 'body_temp',
    'HeartRate': 'heart_rate'
})

# Rename CTG columns for clarity
df_ctg = df_ctg.rename(columns={
    'LB': 'fhr_baseline',       # Fetal Heart Rate baseline
    'AC': 'accelerations',      # Accelerations per second
    'FM': 'fetal_movement',     # Fetal movements per second
    'UC': 'uterine_contractions', # Uterine contractions per second
    'ASTV': 'pct_abnormal_stv', # % time with abnormal short-term variability
    'MSTV': 'mean_stv',         # Mean short-term variability
    'ALTV': 'pct_abnormal_ltv', # % time with abnormal long-term variability
    'MLTV': 'mean_ltv',         # Mean long-term variability
    'DL': 'light_decelerations',
    'DS': 'severe_decelerations',
    'DP': 'prolonged_decelerations',
    'Width': 'histogram_width',
    'Min': 'histogram_min',
    'Max': 'histogram_max',
    'Nmax': 'histogram_peaks',
    'Nzeros': 'histogram_zeros',
    'Mode': 'histogram_mode',
    'Mean': 'histogram_mean',
    'Median': 'histogram_median',
    'Variance': 'histogram_variance',
    'Tendency': 'histogram_tendency'
})

# Build unified feature space:
# Common: heart_rate (mother vs fetal baseline), basic vitals
# Strategy: Use ALL features, fill missing with NaN, then impute

# Maternal → pad missing CTG features with NaN
ctg_only_cols = ['fhr_baseline', 'accelerations', 'fetal_movement',
                 'uterine_contractions', 'pct_abnormal_stv', 'mean_stv',
                 'pct_abnormal_ltv', 'mean_ltv', 'light_decelerations',
                 'severe_decelerations', 'prolonged_decelerations',
                 'histogram_width', 'histogram_min', 'histogram_max',
                 'histogram_peaks', 'histogram_zeros', 'histogram_mode',
                 'histogram_mean', 'histogram_median', 'histogram_variance',
                 'histogram_tendency']

maternal_only_cols = ['age', 'systolic_bp', 'diastolic_bp', 'blood_sugar', 'body_temp']

# Add missing columns as NaN
for col in ctg_only_cols:
    df_maternal[col] = np.nan

for col in maternal_only_cols:
    df_ctg[col] = np.nan

# Map CTG heart rate to common column
df_ctg['heart_rate'] = df_ctg['fhr_baseline']

# ─────────────────────────────────────────────
# 3. MERGE DATASETS
# ─────────────────────────────────────────────

all_features = maternal_only_cols + ['heart_rate'] + ctg_only_cols

df_maternal_aligned = df_maternal[all_features + ['source']]
df_ctg_aligned = df_ctg[all_features + ['source']]

df_combined = pd.concat([df_maternal_aligned, df_ctg_aligned], axis=0, ignore_index=True)

print(f"      [OK] Combined dataset: {df_combined.shape[0]} rows, {len(all_features)} features")
print(f"         Maternal records : {len(df_maternal_aligned)}")
print(f"         CTG records      : {len(df_ctg_aligned)}")

# ─────────────────────────────────────────────
# 4. STABLE SOURCE-AWARE IMPUTATION
# ─────────────────────────────────────────────

print("\n[4/5] Stable Imputation & Feature Engineering...")

df_combined['pulse_pressure'] = df_combined['systolic_bp'] - df_combined['diastolic_bp']
if 'pulse_pressure' not in all_features: all_features.append('pulse_pressure')

# Source-Aware Imputation (Stable)
for source_val in ['maternal', 'ctg']:
    mask = df_combined['source'] == source_val
    for col in all_features:
        if df_combined.loc[mask, col].isnull().all():
            # Fill with global median if entirely missing from source
            df_combined.loc[mask, col] = df_combined[col].median()
        else:
            df_combined.loc[mask, col] = df_combined.loc[mask, col].fillna(df_combined.loc[mask, col].median())

# Final fallback
df_combined[all_features] = df_combined[all_features].fillna(df_combined[all_features].median())
X_imputed = df_combined[all_features].copy()

# ─────────────────────────────────────────────
# 5. SCALE & FEATURE WEIGHTING
# ─────────────────────────────────────────────

from sklearn.preprocessing import StandardScaler
print("\n[5/5] Scaling & Feature Weighting...")

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=all_features)

# UNSUPERVISED TRICK: Weight critical risk features higher
# This tells K-Means: "These features matter more for patient similarity"
risk_weights = {
    'systolic_bp': 2.0,
    'blood_sugar': 2.0,
    'heart_rate': 1.5,
    'pct_abnormal_stv': 2.0,
    'fhr_baseline': 1.5
}

for feat, weight in risk_weights.items():
    if feat in X_scaled.columns:
        X_scaled[feat] = X_scaled[feat] * weight

print(f"      [OK] Preprocessing complete. Shape: {X_scaled.shape}")

# Save processed data
df_combined['source'].to_csv('source_labels.csv', index=False)
X_scaled.to_csv('X_scaled.csv', index=False)
X_imputed.to_csv('X_raw.csv', index=False)

import pickle
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')
imputer.fit(X_imputed)
pickle.dump(scaler, open('scaler.pkl', 'wb'))
pickle.dump(imputer, open('imputer.pkl', 'wb'))

print("\n" + "=" * 60)
print("  [SUCCESS] Step 1 Complete!")
print("  Saved: X_scaled.csv, X_raw.csv, scaler.pkl, imputer.pkl")
print("=" * 60)

# Quick peek
print(f"\nSample of combined data:")
print(X_imputed.describe().round(2))
