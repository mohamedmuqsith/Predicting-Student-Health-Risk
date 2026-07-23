"""
CIS6005 / Kaggle Leaderboard Booster — Student Health Risk Prediction
Generates a Top-Tier Kaggle Leaderboard Submission using LightGBM, XGBoost, and HistGradientBoosting Ensemble.

Run: python run_kaggle_champion.py
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
    print(f"\n{'='*65}\n  {msg}\n{'='*65}", flush=True)

# ── 1. LOAD DATA ──────────────────────────────────────────────
banner("PHASE 1 — Loading Full Raw Dataset")
train_raw = pd.read_csv(RAW / 'train.csv')
test_raw  = pd.read_csv(RAW / 'test.csv')
print(f"  Train shape: {train_raw.shape} | Test shape: {test_raw.shape}")

# ── 2. ADVANCED FEATURE ENGINEERING ───────────────────────────
banner("PHASE 2 — Advanced Kaggle Feature Engineering")

def create_advanced_features(df_input):
    df = df_input.copy()
    
    # 1. Standardize text columns
    obj_cols = [c for c in df.select_dtypes('object').columns if c != 'health_condition']
    for col in obj_cols:
        df[col] = df[col].astype(str).str.lower().str.strip()
    if 'health_condition' in df.columns:
        df['health_condition'] = df['health_condition'].astype(str).str.lower().str.strip()
        
    # Impute missing values
    num_cols = [c for c in df.select_dtypes(['int64','float64']).columns if c != 'id']
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())
    for col in obj_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    # 2. Key Ratios & Interaction Terms
    if 'calorie_expenditure' in df.columns and 'exercise_duration' in df.columns:
        df['activity_efficiency'] = df['calorie_expenditure'] / (df['exercise_duration'] + 1.0)
    if 'step_count' in df.columns and 'calorie_expenditure' in df.columns:
        df['steps_per_calorie'] = df['step_count'] / (df['calorie_expenditure'] + 1.0)
    if 'heart_rate' in df.columns and 'sleep_duration' in df.columns:
        df['heart_rate_to_sleep'] = df['heart_rate'] / (df['sleep_duration'] + 0.5)
    if 'step_count' in df.columns and 'bmi' in df.columns:
        df['steps_per_bmi'] = df['step_count'] / (df['bmi'] + 0.1)
    if 'water_intake' in df.columns and 'exercise_duration' in df.columns:
        df['water_per_exercise'] = df['water_intake'] / (df['exercise_duration'] + 1.0)

    # 3. Domain Medical Rule Indicators
    if 'sleep_duration' in df.columns:
        df['is_ideal_sleep'] = ((df['sleep_duration'] >= 7.0) & (df['sleep_duration'] <= 9.0)).astype(int)
        df['is_deprived_sleep'] = (df['sleep_duration'] < 6.0).astype(int)
    if 'bmi' in df.columns:
        df['is_normal_bmi'] = ((df['bmi'] >= 18.5) & (df['bmi'] <= 24.9)).astype(int)
        df['is_obese'] = (df['bmi'] >= 30.0).astype(int)
    if 'heart_rate' in df.columns:
        df['is_tachycardia'] = (df['heart_rate'] > 100.0).astype(int)
        df['is_bradycardia'] = (df['heart_rate'] < 60.0).astype(int)
    if 'step_count' in df.columns:
        df['is_active_steps'] = (df['step_count'] >= 8000).astype(int)

    return df

df_full = pd.concat([train_raw.assign(is_train=1), test_raw.assign(is_train=0, health_condition=np.nan)], ignore_index=True)
df_full = create_advanced_features(df_full)

# Group Aggregation Features (Statistical baselines across gender & physical activity level)
group_cols = ['gender', 'physical_activity_level']
for agg_col in ['heart_rate', 'bmi', 'step_count', 'calorie_expenditure']:
    if agg_col in df_full.columns:
        group_mean = df_full.groupby(group_cols)[agg_col].transform('mean')
        df_full[f'{agg_col}_diff_from_group_mean'] = df_full[agg_col] - group_mean

train_proc = df_full[df_full['is_train'] == 1].drop(columns=['is_train'])
test_proc  = df_full[df_full['is_train'] == 0].drop(columns=['is_train'])

print(f"  Total Engineered Features: {train_proc.shape[1] - 2}")

# ── 3. ENCODING & PREPROCESSING ───────────────────────────────
banner("PHASE 3 — Target & Categorical Encoding")

TARGET = 'health_condition'
le_target = LabelEncoder()
y_all = le_target.fit_transform(train_proc[TARGET])
CLASS_NAMES = list(le_target.classes_)
print(f"  Target Classes: {dict(zip(CLASS_NAMES, range(len(CLASS_NAMES))))}")

X_all = train_proc.drop(columns=['id', TARGET], errors='ignore')
test_ids = test_proc['id'] if 'id' in test_proc.columns else pd.Series(range(len(test_proc)))
X_test = test_proc.drop(columns=['id', TARGET], errors='ignore')

# Categorical Frequency Encoding & Label Encoding
cat_encoders = {}
cat_cols = X_all.select_dtypes('object').columns.tolist()
for c in cat_cols:
    le = LabelEncoder()
    # Fit on combined to avoid unseen categories
    combined_cats = pd.concat([X_all[c].astype(str), X_test[c].astype(str)])
    le.fit(combined_cats)
    X_all[c] = le.transform(X_all[c].astype(str))
    X_test[c] = le.transform(X_test[c].astype(str))
    cat_encoders[c] = le

# Scale features
scaler = StandardScaler()
X_all_scaled = pd.DataFrame(scaler.fit_transform(X_all), columns=X_all.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_all.columns)

# ── 4. KAGGLE CHAMPION MODEL TRAINING (ENSEMBLE) ──────────────
banner("PHASE 4 — Training Top Kaggle GBDT Ensemble (150K Stratified Sample)")

# Sample 150K for optimal training speed & accuracy
SAMPLE_SIZE = min(150_000, len(X_all_scaled))
_, X_sub, _, y_sub = train_test_split(
    X_all_scaled, y_all,
    test_size=SAMPLE_SIZE/len(X_all_scaled),
    random_state=RS, stratify=y_all
)

X_tr, X_val, y_tr, y_val = train_test_split(
    X_sub, y_sub,
    test_size=0.20, random_state=RS, stratify=y_sub
)
sample_weights_tr = compute_sample_weight('balanced', y_tr)

print(f"  Training on {len(X_tr):,} samples, Validating on {len(X_val):,} samples")

# Model 1: LightGBM
print("  [1/3] Training LightGBM Classifier...", flush=True)
model_lgb = lgb.LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=6,
    class_weight='balanced',
    random_state=RS,
    n_jobs=-1,
    verbose=-1
)
t0 = time.time()
model_lgb.fit(X_tr, y_tr)
lgb_val_preds_proba = model_lgb.predict_proba(X_val)
lgb_test_preds_proba = model_lgb.predict_proba(X_test_scaled)
print(f"        LightGBM completed in {time.time()-t0:.1f}s")

# Model 2: XGBoost
print("  [2/3] Training XGBoost Classifier...", flush=True)
model_xgb = xgb.XGBClassifier(
    n_estimators=250,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=RS,
    n_jobs=-1,
    verbosity=0
)
t0 = time.time()
model_xgb.fit(X_tr, y_tr, sample_weight=sample_weights_tr)
xgb_val_preds_proba = model_xgb.predict_proba(X_val)
xgb_test_preds_proba = model_xgb.predict_proba(X_test_scaled)
print(f"        XGBoost completed in {time.time()-t0:.1f}s")

# Model 3: HistGradientBoosting
print("  [3/3] Training HistGradientBoosting Classifier...", flush=True)
model_hgb = HistGradientBoostingClassifier(
    max_iter=250,
    learning_rate=0.05,
    max_depth=6,
    class_weight='balanced',
    random_state=RS
)
t0 = time.time()
model_hgb.fit(X_tr, y_tr)
hgb_val_preds_proba = model_hgb.predict_proba(X_val)
hgb_test_preds_proba = model_hgb.predict_proba(X_test_scaled)
print(f"        HistGradientBoosting completed in {time.time()-t0:.1f}s")

# ── 5. BLENDING & ENSEMBLE EVALUATION ─────────────────────────
banner("PHASE 5 — Ensemble Soft-Voting & Performance Metrics")

# Weighted average probabilities (LightGBM: 40%, XGBoost: 40%, HistGB: 20%)
ensemble_val_proba  = (0.40 * lgb_val_preds_proba) + (0.40 * xgb_val_preds_proba) + (0.20 * hgb_val_preds_proba)
ensemble_test_proba = (0.40 * lgb_test_preds_proba) + (0.40 * xgb_test_preds_proba) + (0.20 * hgb_test_preds_proba)

val_preds = np.argmax(ensemble_val_proba, axis=1)

val_acc  = accuracy_score(y_val, val_preds)
val_f1_w = f1_score(y_val, val_preds, average='weighted')
val_f1_m = f1_score(y_val, val_preds, average='macro')

print(f"  Ensemble Validation Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
print(f"  Ensemble F1 Score (Weighted): {val_f1_w:.4f}")
print(f"  Ensemble F1 Score (Macro)   : {val_f1_m:.4f}")
print()
print(classification_report(y_val, val_preds, target_names=CLASS_NAMES))

# ── 6. GENERATE KAGGLE SUBMISSION ─────────────────────────────
banner("PHASE 6 — Generating High-Rank Kaggle Submission")

test_preds = np.argmax(ensemble_test_proba, axis=1)
test_preds_labels = le_target.inverse_transform(test_preds)

sub_df = pd.DataFrame({
    'id': test_ids.values,
    'health_condition': test_preds_labels
})

ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
submission_name = f'submission_Kaggle_Champion_Ensemble_{ts}.csv'
submission_path = SUBMIT / submission_name
sub_df.to_csv(submission_path, index=False)

print("  Class Prediction Breakdown:")
u, c = np.unique(test_preds_labels, return_counts=True)
for cls, cnt in zip(u, c):
    print(f"    {cls:<15}: {cnt:>7,}  ({cnt/len(test_preds_labels)*100:.1f}%)")

print()
print(f"  🎉 SUBMISSION FILE SAVED SUCCESSFULLY:")
print(f"     Path: {submission_path}")

# ── 7. UPDATE PRODUCTION BUNDLE ──────────────────────────────
bundle = {
    'model': model_lgb,
    'ensemble_models': [model_lgb, model_xgb, model_hgb],
    'scaler': scaler,
    'label_encoder': le_target,
    'categorical_encoders': cat_encoders,
    'feature_names': list(X_all.columns),
    'class_names': CLASS_NAMES,
    'val_accuracy': round(val_acc, 4),
    'val_f1_weighted': round(val_f1_w, 4),
    'project': 'CIS6005 Student Health Risk Prediction - Kaggle Champion'
}
joblib.dump(bundle, MODELS / 'production_bundle.joblib')
print("  Updated models/production_bundle.joblib [OK]")
banner("KAGGLE CHAMPION PIPELINE COMPLETE")
