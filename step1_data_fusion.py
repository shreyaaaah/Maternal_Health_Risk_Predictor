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

# ─────────────────────────────────────────────
# 1. LOAD DATASETS
# ─────────────────────────────────────────────

print("\n[1/5] Loading UCI Maternal Health Risk dataset...")
maternal = fetch_ucirepo(id=863)
df_maternal = maternal.data.features.copy()
df_maternal['source'] = 'maternal'
print(f"      ✅ Loaded: {df_maternal.shape[0]} rows, {df_maternal.shape[1]-1} features")
print(f"      Features: {list(df_maternal.columns[:-1])}")

print("\n[2/5] Loading UCI CTG (Cardiotocography) dataset...")
ctg = fetch_ucirepo(id=193)
df_ctg = ctg.data.features.copy()
df_ctg['source'] = 'ctg'
print(f"      ✅ Loaded: {df_ctg.shape[0]} rows, {df_ctg.shape[1]-1} features")
print(f"      Features: {list(df_ctg.columns[:10])}... (21 total)")

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

print(f"      ✅ Combined dataset: {df_combined.shape[0]} rows, {len(all_features)} features")
print(f"         Maternal records : {len(df_maternal_aligned)}")
print(f"         CTG records      : {len(df_ctg_aligned)}")

# ─────────────────────────────────────────────
# 4. HANDLE MISSING VALUES
# ─────────────────────────────────────────────

print("\n[4/5] Handling missing values...")

X = df_combined[all_features].copy()

# Fill NaN with column median (safe for skewed data)
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=all_features)

missing_before = df_combined[all_features].isnull().sum().sum()
print(f"      Missing values before imputation: {missing_before}")
print(f"      Missing values after imputation : {X_imputed.isnull().sum().sum()}")

# ─────────────────────────────────────────────
# 5. SCALE FEATURES
# ─────────────────────────────────────────────

print("\n[5/5] Scaling features (StandardScaler)...")
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=all_features)
print(f"      ✅ Scaling complete. Shape: {X_scaled.shape}")

# Save processed data
df_combined['source'].to_csv('source_labels.csv', index=False)
X_scaled.to_csv('X_scaled.csv', index=False)
X_imputed.to_csv('X_raw.csv', index=False)

import pickle
pickle.dump(scaler, open('scaler.pkl', 'wb'))
pickle.dump(imputer, open('imputer.pkl', 'wb'))

print("\n" + "=" * 60)
print("  ✅ Step 1 Complete!")
print("  Saved: X_scaled.csv, X_raw.csv, scaler.pkl, imputer.pkl")
print("=" * 60)

# Quick peek
print(f"\nSample of combined data:")
print(X_imputed.describe().round(2))
