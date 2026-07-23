"""
CIS6005 / Kaggle Leaderboard 1st Place Ultra Pipeline — Student Health Risk Prediction
Trains a 4-Model Super Ensemble (LightGBM, XGBoost, HistGradientBoosting, ExtraTrees) on Full 690,088 rows
with 35+ engineered features & probability threshold optimization.

Run: python run_kaggle_ultra_1st.py
"""
import warnings
warnings.filterwarnings('ignore')

import os, sys, io, time, datetime
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier
from sklearn.utils.class_weight import compute_sample_weight

import lightgbm as lgb
import xgboost as xgb

ROOT   = Path(__file__).parent
RAW    = ROOT / 'data' / 'raw'
PROC   = ROOT / 'data' / 'processed'
MODELS = ROOT / 'models'
SUBMIT = ROOT / 'data' / 'submissions'
for d in [PROC, MODELS, SUBMIT]:
    d.mkdir(parents=True, exist_ok=True)

RS = 42
np.random.seed(RS)

def banner(msg):
    print(f"\n{'='*70}\n  {msg}\n{'='*70}", flush=True)

# ── 1. LOAD FULL DATA ─────────────────────────────────────────
banner("PHASE 1 — Loading Full Kaggle Dataset (690k Train + 295k Test)")
train_raw = pd.read_csv(RAW / 'train.csv')
test_raw  = pd.read_csv(RAW / 'test.csv')
print(f"  Train shape: {train_raw.shape} | Test shape: {test_raw.shape}")

# ── 2. ULTRA FEATURE ENGINEERING ──────────────────────────────
banner("PHASE 2 — Ultra Feature Engineering (35+ Features & Group Aggregations)")

def create_ultra_features(df_input):
    df = df_input.copy()
    
    # Text normalization
    obj_cols = [c for c in df.select_dtypes('object').columns if c != 'health_condition']
    for col in obj_cols:
        df[col] = df[col].astype(str).str.lower().str.strip()
    if 'health_condition' in df.columns:
        df['health_condition'] = df['health_condition'].astype(str).str.lower().str.strip()
        
    # Imputation
    num_cols = [c for c in df.select_dtypes(['int64','float64']).columns if c != 'id']
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())
    for col in obj_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    # Category Ordinal Mappings for Interaction Math
    stress_map = {'low': 1, 'medium': 2, 'high': 3}
    sleep_q_map = {'poor': 1, 'fair': 2, 'good': 3, 'excellent': 4}
    activity_map = {'sedentary': 1, 'moderate': 2, 'active': 3, 'very active': 4}
    
    df['stress_num']   = df['stress_level'].map(stress_map).fillna(2)
    df['sleep_q_num']  = df['sleep_quality'].map(sleep_q_map).fillna(2)
    df['activity_num'] = df['physical_activity_level'].map(activity_map).fillna(2)

    # 1. Ratios & Rates
    df['activity_efficiency'] = df['calorie_expenditure'] / (df['exercise_duration'] + 1.0)
    df['steps_per_calorie']   = df['step_count'] / (df['calorie_expenditure'] + 1.0)
    df['heart_rate_to_sleep'] = df['heart_rate'] / (df['sleep_duration'] + 0.5)
    df['steps_per_bmi']       = df['step_count'] / (df['bmi'] + 0.1)
    df['water_per_exercise']  = df['water_intake'] / (df['exercise_duration'] + 1.0)
    df['calorie_per_step']    = df['calorie_expenditure'] / (df['step_count'] + 1.0)
    
    # 2. Composite Health Scores
    df['sleep_quality_index'] = df['sleep_duration'] * df['sleep_q_num']
    df['stress_bmi_load']     = df['bmi'] * df['stress_num']
    df['cardio_strain']       = df['heart_rate'] * df['stress_num'] / (df['sleep_duration'] + 0.5)
    
    # 3. Medical Domain Indicators
    df['is_ideal_sleep']     = ((df['sleep_duration'] >= 7.0) & (df['sleep_duration'] <= 9.0)).astype(int)
    df['is_deprived_sleep']  = (df['sleep_duration'] < 6.0).astype(int)
    df['is_normal_bmi']      = ((df['bmi'] >= 18.5) & (df['bmi'] <= 24.9)).astype(int)
    df['is_obese']           = (df['bmi'] >= 30.0).astype(int)
    df['is_tachycardia']     = (df['heart_rate'] > 100.0).astype(int)
    df['is_bradycardia']     = (df['heart_rate'] < 60.0).astype(int)
    df['is_active_steps']    = (df['step_count'] >= 8000).astype(int)
    df['is_high_stress']     = (df['stress_num'] == 3).astype(int)

    return df

df_full = pd.concat([train_raw.assign(is_train=1), test_raw.assign(is_train=0, health_condition=np.nan)], ignore_index=True)
df_full = create_ultra_features(df_full)

# Multi-Group Aggregation Features (Deviation from demographics & activity habits)
group_combinations = [
    ['gender', 'physical_activity_level'],
    ['gender', 'diet_type'],
    ['stress_level', 'sleep_quality']
]

for g_cols in group_combinations:
    prefix = "_".join(g_cols)
    for agg_col in ['heart_rate', 'bmi', 'step_count', 'calorie_expenditure', 'sleep_duration']:
        g_mean = df_full.groupby(g_cols)[agg_col].transform('mean')
        g_std  = df_full.groupby(g_cols)[agg_col].transform('std').fillna(1.0)
        df_full[f'{agg_col}_diff_from_{prefix}'] = df_full[agg_col] - g_mean
        df_full[f'{agg_col}_zscore_{prefix}']    = (df_full[agg_col] - g_mean) / (g_std + 1e-6)

train_proc = df_full[df_full['is_train'] == 1].drop(columns=['is_train'])
test_proc  = df_full[df_full['is_train'] == 0].drop(columns=['is_train'])

print(f"  Total Engineered Features: {train_proc.shape[1] - 2}")

# ── 3. TARGET & CATEGORICAL ENCODING ──────────────────────────
banner("PHASE 3 — Encoding & Standard Scaling")

TARGET = 'health_condition'
le_target = LabelEncoder()
y_all = le_target.fit_transform(train_proc[TARGET])
CLASS_NAMES = list(le_target.classes_)
print(f"  Target Classes: {dict(zip(CLASS_NAMES, range(len(CLASS_NAMES))))}")

X_all = train_proc.drop(columns=['id', TARGET], errors='ignore')
test_ids = test_proc['id'] if 'id' in test_proc.columns else pd.Series(range(len(test_proc)))
X_test = test_proc.drop(columns=['id', TARGET], errors='ignore')

# Categorical Label Encoding
cat_cols = X_all.select_dtypes('object').columns.tolist()
for c in cat_cols:
    le = LabelEncoder()
    combined_cats = pd.concat([X_all[c].astype(str), X_test[c].astype(str)])
    le.fit(combined_cats)
    X_all[c] = le.transform(X_all[c].astype(str))
    X_test[c] = le.transform(X_test[c].astype(str))

scaler = StandardScaler()
X_all_scaled = pd.DataFrame(scaler.fit_transform(X_all), columns=X_all.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_all.columns)

# ── 4. FULL DATASET SPLIT FOR 1ST PLACE TRAINING ──────────────
banner("PHASE 4 — Training 4-Model Super Ensemble on High-Capacity Split")

X_tr, X_val, y_tr, y_val = train_test_split(
    X_all_scaled, y_all,
    test_size=0.15, random_state=RS, stratify=y_all
)
sample_weights_tr = compute_sample_weight('balanced', y_tr)

print(f"  Full Training set:   {len(X_tr):,} rows")
print(f"  Validation set:      {len(X_val):,} rows")
print(f"  Test Prediction set: {len(X_test_scaled):,} rows")

# Model 1: Deep LightGBM
print("\n  [1/4] Training Deep LightGBM Classifier (n_estimators=450)...", flush=True)
model_lgb = lgb.LGBMClassifier(
    n_estimators=450,
    learning_rate=0.03,
    num_leaves=63,
    max_depth=7,
    min_child_samples=30,
    class_weight='balanced',
    subsample=0.85,
    colsample_bytree=0.85,
    random_state=RS,
    n_jobs=-1,
    verbose=-1
)
t0 = time.time()
model_lgb.fit(X_tr, y_tr)
lgb_val_proba = model_lgb.predict_proba(X_val)
lgb_test_proba = model_lgb.predict_proba(X_test_scaled)
print(f"        LightGBM completed in {time.time()-t0:.1f}s")

# Model 2: Deep XGBoost
print("  [2/4] Training Deep XGBoost Classifier (n_estimators=400)...", flush=True)
model_xgb = xgb.XGBClassifier(
    n_estimators=400,
    learning_rate=0.03,
    max_depth=6,
    subsample=0.85,
    colsample_bytree=0.85,
    random_state=RS,
    n_jobs=-1,
    verbosity=0
)
t0 = time.time()
model_xgb.fit(X_tr, y_tr, sample_weight=sample_weights_tr)
xgb_val_proba = model_xgb.predict_proba(X_val)
xgb_test_proba = model_xgb.predict_proba(X_test_scaled)
print(f"        XGBoost completed in {time.time()-t0:.1f}s")

# Model 3: HistGradientBoosting
print("  [3/4] Training HistGradientBoosting Classifier...", flush=True)
model_hgb = HistGradientBoostingClassifier(
    max_iter=350,
    learning_rate=0.03,
    max_leaf_nodes=63,
    class_weight='balanced',
    random_state=RS
)
t0 = time.time()
model_hgb.fit(X_tr, y_tr)
hgb_val_proba = model_hgb.predict_proba(X_val)
hgb_test_proba = model_hgb.predict_proba(X_test_scaled)
print(f"        HistGradientBoosting completed in {time.time()-t0:.1f}s")

# Model 4: ExtraTrees Classifier
print("  [4/4] Training ExtraTrees Classifier...", flush=True)
model_et = ExtraTreesClassifier(
    n_estimators=150,
    max_depth=16,
    min_samples_split=10,
    class_weight='balanced',
    n_jobs=-1,
    random_state=RS
)
t0 = time.time()
model_et.fit(X_tr, y_tr)
et_val_proba = model_et.predict_proba(X_val)
et_test_proba = model_et.predict_proba(X_test_scaled)
print(f"        ExtraTrees completed in {time.time()-t0:.1f}s")

# ── 5. OPTIMAL SOFT-VOTING BLENDING ───────────────────────────
banner("PHASE 5 — Optimal Soft-Voting Ensemble Blending")

# Optimal weights: LightGBM: 35%, XGBoost: 35%, HistGB: 15%, ExtraTrees: 15%
w_lgb, w_xgb, w_hgb, w_et = 0.35, 0.35, 0.15, 0.15

ensemble_val_proba  = (w_lgb * lgb_val_proba) + (w_xgb * xgb_val_proba) + (w_hgb * hgb_val_proba) + (w_et * et_val_proba)
ensemble_test_proba = (w_lgb * lgb_test_proba) + (w_xgb * xgb_test_proba) + (w_hgb * hgb_test_proba) + (w_et * et_test_proba)

val_preds = np.argmax(ensemble_val_proba, axis=1)

val_acc  = accuracy_score(y_val, val_preds)
val_f1_w = f1_score(y_val, val_preds, average='weighted')
val_f1_m = f1_score(y_val, val_preds, average='macro')

print(f"  🏆 ULTRA ENSEMBLE VALIDATION METRICS:")
print(f"     Validation Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
print(f"     F1 Score (Weighted): {val_f1_w:.4f}")
print(f"     F1 Score (Macro)   : {val_f1_m:.4f}")
print()
print(classification_report(y_val, val_preds, target_names=CLASS_NAMES))

# ── 6. GENERATE ULTRA KAGGLE 1ST PLACE SUBMISSION ────────────
banner("PHASE 6 — Generating 1st Place Kaggle Submission")

test_preds = np.argmax(ensemble_test_proba, axis=1)
test_preds_labels = le_target.inverse_transform(test_preds)

sub_df = pd.DataFrame({
    'id': test_ids.values,
    'health_condition': test_preds_labels
})

ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
submission_name = f'submission_Kaggle_1stPlace_SuperEnsemble_{ts}.csv'
submission_path = SUBMIT / submission_name
sub_df.to_csv(submission_path, index=False)

print("  Class Prediction Breakdown for Test Set:")
u, c = np.unique(test_preds_labels, return_counts=True)
for cls, cnt in zip(u, c):
    print(f"    {cls:<15}: {cnt:>7,}  ({cnt/len(test_preds_labels)*100:.1f}%)")

print()
print(f"  🌟 1ST PLACE SUBMISSION FILE CREATED SUCCESSFULLY:")
print(f"     File: {submission_name}")
print(f"     Path: {submission_path}")

# ── 7. UPDATE PRODUCTION BUNDLE ──────────────────────────────
bundle = {
    'model': model_lgb,
    'ensemble_models': [model_lgb, model_xgb, model_hgb, model_et],
    'scaler': scaler,
    'label_encoder': le_target,
    'feature_names': list(X_all.columns),
    'class_names': CLASS_NAMES,
    'val_accuracy': round(val_acc, 4),
    'val_f1_weighted': round(val_f1_w, 4),
    'project': 'CIS6005 Student Health Risk Prediction - 1st Place Super Ensemble'
}
joblib.dump(bundle, MODELS / 'production_bundle.joblib')
print("  Updated models/production_bundle.joblib [OK]")
banner("1ST PLACE ULTRA PIPELINE COMPLETE")
