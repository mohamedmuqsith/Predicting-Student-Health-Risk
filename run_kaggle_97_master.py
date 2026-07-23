"""
CIS6005 / Kaggle 97+ Target Master Pipeline — Student Health Risk Prediction
Uses Fast High-Capacity Triad Ensemble (LightGBM, XGBoost, HistGB) across 690,088 rows
with 87 Master Features & Scipy Multiplier Threshold Optimization.

Run: python run_kaggle_97_master.py
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight
from scipy.optimize import minimize

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
    print(f"\n{'='*75}\n  {msg}\n{'='*75}", flush=True)

# ── 1. LOAD FULL DATA ─────────────────────────────────────────
banner("PHASE 1 — Loading Full Raw Dataset (690k Train + 295k Test)")
train_raw = pd.read_csv(RAW / 'train.csv')
test_raw  = pd.read_csv(RAW / 'test.csv')
print(f"  Train shape: {train_raw.shape} | Test shape: {test_raw.shape}")

# ── 2. MASTER FEATURE ENGINEERING (87 FEATURES) ───────────────
banner("PHASE 2 — Master Feature Engineering (Non-Linear & Multi-Level Group Aggregations)")

def create_master_features(df_input):
    df = df_input.copy()
    
    obj_cols = [c for c in df.select_dtypes('object').columns if c != 'health_condition']
    for col in obj_cols:
        df[col] = df[col].astype(str).str.lower().str.strip()
    if 'health_condition' in df.columns:
        df['health_condition'] = df['health_condition'].astype(str).str.lower().str.strip()
        
    num_cols = [c for c in df.select_dtypes(['int64','float64']).columns if c != 'id']
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())
    for col in obj_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    stress_map   = {'low': 1, 'medium': 2, 'high': 3}
    sleep_q_map  = {'poor': 1, 'fair': 2, 'good': 3, 'excellent': 4}
    activity_map = {'sedentary': 1, 'moderate': 2, 'active': 3, 'very active': 4}
    smoke_map    = {'no': 0, 'yes': 1}
    
    df['stress_num']   = df['stress_level'].map(stress_map).fillna(2)
    df['sleep_q_num']  = df['sleep_quality'].map(sleep_q_map).fillna(2)
    df['activity_num'] = df['physical_activity_level'].map(activity_map).fillna(2)
    df['smoke_num']    = df['smoking_alcohol'].map(smoke_map).fillna(0)

    # 1. Ratios & Interactions
    df['activity_efficiency']  = df['calorie_expenditure'] / (df['exercise_duration'] + 1.0)
    df['steps_per_calorie']    = df['step_count'] / (df['calorie_expenditure'] + 1.0)
    df['heart_rate_to_sleep']  = df['heart_rate'] / (df['sleep_duration'] + 0.5)
    df['steps_per_bmi']        = df['step_count'] / (df['bmi'] + 0.1)
    df['water_per_exercise']   = df['water_intake'] / (df['exercise_duration'] + 1.0)
    df['calorie_per_step']     = df['calorie_expenditure'] / (df['step_count'] + 1.0)
    
    # 2. Non-linear transformations
    df['bmi_squared']          = df['bmi'] ** 2
    df['heart_rate_squared']   = df['heart_rate'] ** 2
    df['sleep_duration_sq']    = df['sleep_duration'] ** 2
    df['step_count_log']       = np.log1p(df['step_count'])

    # 3. Composite Health Indices
    df['sleep_quality_index']  = df['sleep_duration'] * df['sleep_q_num']
    df['stress_bmi_load']      = df['bmi'] * df['stress_num']
    df['cardio_strain']        = (df['heart_rate'] * df['stress_num']) / (df['sleep_duration'] + 0.5)
    df['unhealth_risk_index']  = (df['bmi'] * 0.4) + (df['heart_rate'] * 0.3) + (df['stress_num'] * 5.0) + (df['smoke_num'] * 10.0) - (df['sleep_quality_index'] * 2.0) - (df['step_count'] / 1000.0)

    # 4. Medical Rules
    df['is_ideal_sleep']      = ((df['sleep_duration'] >= 7.0) & (df['sleep_duration'] <= 9.0)).astype(int)
    df['is_deprived_sleep']   = (df['sleep_duration'] < 6.0).astype(int)
    df['is_normal_bmi']       = ((df['bmi'] >= 18.5) & (df['bmi'] <= 24.9)).astype(int)
    df['is_obese']            = (df['bmi'] >= 30.0).astype(int)
    df['is_tachycardia']      = (df['heart_rate'] > 100.0).astype(int)
    df['is_bradycardia']      = (df['heart_rate'] < 60.0).astype(int)
    df['is_active_steps']     = (df['step_count'] >= 8000).astype(int)
    df['is_high_stress']      = (df['stress_num'] == 3).astype(int)

    return df

df_full = pd.concat([train_raw.assign(is_train=1), test_raw.assign(is_train=0, health_condition=np.nan)], ignore_index=True)
df_full = create_master_features(df_full)

# Multi-level Demographic Group Aggregations
group_combos = [
    ['gender', 'physical_activity_level'],
    ['gender', 'diet_type'],
    ['stress_level', 'sleep_quality'],
    ['gender', 'smoking_alcohol']
]

for g_cols in group_combos:
    prefix = "_".join(g_cols)
    for agg_col in ['heart_rate', 'bmi', 'step_count', 'calorie_expenditure', 'sleep_duration', 'unhealth_risk_index']:
        g_mean = df_full.groupby(g_cols)[agg_col].transform('mean')
        g_std  = df_full.groupby(g_cols)[agg_col].transform('std').fillna(1.0)
        df_full[f'{agg_col}_diff_{prefix}']  = df_full[agg_col] - g_mean
        df_full[f'{agg_col}_zscore_{prefix}'] = (df_full[agg_col] - g_mean) / (g_std + 1e-6)

train_proc = df_full[df_full['is_train'] == 1].drop(columns=['is_train'])
test_proc  = df_full[df_full['is_train'] == 0].drop(columns=['is_train'])

print(f"  Total Master Engineered Features: {train_proc.shape[1] - 2}")

# ── 3. TARGET & CATEGORICAL ENCODING ──────────────────────────
banner("PHASE 3 — Encoding & Standard Scaling")

TARGET = 'health_condition'
le_target = LabelEncoder()
y_all = le_target.fit_transform(train_proc[TARGET])
CLASS_NAMES = list(le_target.classes_)
N_CLASSES = len(CLASS_NAMES)
print(f"  Target Classes: {dict(zip(CLASS_NAMES, range(N_CLASSES)))}")

X_all = train_proc.drop(columns=['id', TARGET], errors='ignore')
test_ids = test_proc['id'] if 'id' in test_proc.columns else pd.Series(range(len(test_proc)))
X_test = test_proc.drop(columns=['id', TARGET], errors='ignore')

# Categorical Encoding
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

# ── 4. FULL DATASET HIGH-CAPACITY MODEL TRAINING ──────────────
banner("PHASE 4 — Training Master Triad Ensemble on 586,574 rows")

X_tr, X_val, y_tr, y_val = train_test_split(
    X_all_scaled, y_all,
    test_size=0.15, random_state=RS, stratify=y_all
)
sample_weights_tr = compute_sample_weight('balanced', y_tr)

print(f"  Train set:      {len(X_tr):,} rows")
print(f"  Validation set: {len(X_val):,} rows")

# Model 1: LightGBM
print("\n  [1/3] Training Ultra LightGBM (450 trees, depth 7)...", flush=True)
mdl_lgb = lgb.LGBMClassifier(
    n_estimators=450, learning_rate=0.035, num_leaves=63, max_depth=7,
    min_child_samples=30, class_weight='balanced', subsample=0.85,
    colsample_bytree=0.85, random_state=RS, n_jobs=-1, verbose=-1
)
t0 = time.time()
mdl_lgb.fit(X_tr, y_tr)
lgb_val_proba = mdl_lgb.predict_proba(X_val)
lgb_test_proba = mdl_lgb.predict_proba(X_test_scaled)
print(f"        LightGBM completed in {time.time()-t0:.1f}s")

# Model 2: XGBoost
print("  [2/3] Training Ultra XGBoost (400 trees, depth 6)...", flush=True)
mdl_xgb = xgb.XGBClassifier(
    n_estimators=400, learning_rate=0.035, max_depth=6, subsample=0.85,
    colsample_bytree=0.85, random_state=RS, n_jobs=-1, verbosity=0
)
t0 = time.time()
mdl_xgb.fit(X_tr, y_tr, sample_weight=sample_weights_tr)
xgb_val_proba = mdl_xgb.predict_proba(X_val)
xgb_test_proba = mdl_xgb.predict_proba(X_test_scaled)
print(f"        XGBoost completed in {time.time()-t0:.1f}s")

# Model 3: HistGradientBoosting
print("  [3/3] Training HistGradientBoosting (350 trees)...", flush=True)
mdl_hgb = HistGradientBoostingClassifier(
    max_iter=350, learning_rate=0.035, max_leaf_nodes=63,
    class_weight='balanced', random_state=RS
)
t0 = time.time()
mdl_hgb.fit(X_tr, y_tr)
hgb_val_proba = mdl_hgb.predict_proba(X_val)
hgb_test_proba = mdl_hgb.predict_proba(X_test_scaled)
print(f"        HistGB completed in {time.time()-t0:.1f}s")

# ── 5. SCIPY PROBABILITY THRESHOLD OPTIMIZER ──────────────────
banner("PHASE 5 — Scipy Multiplier Threshold Optimization")

raw_val_ensemble  = (0.45 * lgb_val_proba) + (0.45 * xgb_val_proba) + (0.10 * hgb_val_proba)
raw_test_ensemble = (0.45 * lgb_test_proba) + (0.45 * xgb_test_proba) + (0.10 * hgb_test_proba)

raw_val_preds = np.argmax(raw_val_ensemble, axis=1)
raw_acc  = accuracy_score(y_val, raw_val_preds)
raw_f1_w = f1_score(y_val, raw_val_preds, average='weighted')

print(f"  Unoptimized Ensemble Acc : {raw_acc:.4f} ({raw_acc*100:.2f}%)")
print(f"  Unoptimized Weighted F1  : {raw_f1_w:.4f}")

print("\n  Optimizing Threshold Multipliers via Nelder-Mead Optimization...", flush=True)

def loss_func(weights):
    w_probs = raw_val_ensemble * weights
    preds   = np.argmax(w_probs, axis=1)
    return -f1_score(y_val, preds, average='weighted')

init_weights = [1.0, 1.0, 1.0]
res = minimize(loss_func, init_weights, method='Nelder-Mead', options={'maxiter': 500})
best_weights = res.x / np.sum(res.x)

print(f"  Optimal Class Multipliers: {dict(zip(CLASS_NAMES, np.round(best_weights, 4)))}")

opt_val_probs = raw_val_ensemble * best_weights
opt_val_preds = np.argmax(opt_val_probs, axis=1)

opt_acc  = accuracy_score(y_val, opt_val_preds)
opt_f1_w = f1_score(y_val, opt_val_preds, average='weighted')
opt_f1_m = f1_score(y_val, opt_val_preds, average='macro')

print(f"\n  🚀 OPTIMIZED 97 TARGET METRICS:")
print(f"     Validation Accuracy : {opt_acc:.4f} ({opt_acc*100:.2f}%)")
print(f"     F1 Score (Weighted) : {opt_f1_w:.4f}")
print(f"     F1 Score (Macro)    : {opt_f1_m:.4f}")
print()
print(classification_report(y_val, opt_val_preds, target_names=CLASS_NAMES))

# ── 6. GENERATE 97 MASTER KAGGLE SUBMISSION ────────────────────
banner("PHASE 6 — Generating 97 Master Kaggle Submission")

opt_test_probs  = raw_test_ensemble * best_weights
opt_test_preds  = np.argmax(opt_test_probs, axis=1)
opt_test_labels = le_target.inverse_transform(opt_test_preds)

sub_df = pd.DataFrame({
    'id': test_ids.values,
    'health_condition': opt_test_labels
})

ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
submission_name = f'submission_Kaggle_97_Master_Ensemble_{ts}.csv'
submission_path = SUBMIT / submission_name
sub_df.to_csv(submission_path, index=False)

print("  Class Prediction Breakdown for Test Set:")
u, c = np.unique(opt_test_labels, return_counts=True)
for cls, cnt in zip(u, c):
    print(f"    {cls:<15}: {cnt:>7,}  ({cnt/len(opt_test_labels)*100:.1f}%)")

print()
print(f"  🔥 97 MASTER SUBMISSION FILE CREATED SUCCESSFULLY:")
print(f"     File: {submission_name}")
print(f"     Path: {submission_path}")

# ── 7. UPDATE PRODUCTION BUNDLE ──────────────────────────────
bundle = {
    'model': mdl_lgb,
    'ensemble_models': [mdl_lgb, mdl_xgb, mdl_hgb],
    'scaler': scaler,
    'label_encoder': le_target,
    'optimal_weights': best_weights,
    'feature_names': list(X_all.columns),
    'class_names': CLASS_NAMES,
    'val_accuracy': round(opt_acc, 4),
    'val_f1_weighted': round(opt_f1_w, 4),
    'project': 'CIS6005 Student Health Risk Prediction - 97 Master Ensemble'
}
joblib.dump(bundle, MODELS / 'production_bundle.joblib')
print("  Updated models/production_bundle.joblib [OK]")
banner("97 MASTER KAGGLE PIPELINE COMPLETE")
