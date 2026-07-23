"""
train_production_bundle.py
Master Production Model Builder & Integrity Auditor
Student Health Risk Prediction (CIS6005 / Kaggle PS S6E7)
"""

import os
import sys
import io
import hashlib
import json
import datetime
import numpy as np
import pandas as pd
import joblib
import sklearn
from pathlib import Path

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

ROOT = Path(__file__).parent
RAW = ROOT / 'data' / 'raw'
MODELS = ROOT / 'models'
MODELS.mkdir(parents=True, exist_ok=True)

RS = 42
np.random.seed(RS)

print("=" * 65)
print("  MASTER PRODUCTION MODEL TRAINER & INTEGRITY AUDITOR")
print("=" * 65)

# 1. Load Training Data
train_path = RAW / 'train.csv'
if not train_path.exists():
    raise FileNotFoundError(f"Missing {train_path}. Cannot train production model.")

df_raw = pd.read_csv(train_path)
print(f"  Loaded raw dataset: {df_raw.shape[0]:,} rows, {df_raw.shape[1]} columns")

# 2. String Normalisation
df = df_raw.copy()
TARGET = 'health_condition'

obj_cols = [c for c in df.select_dtypes('object').columns if c != TARGET]
for col in obj_cols:
    df[col] = df[col].astype(str).str.lower().str.strip()
df[TARGET] = df[TARGET].astype(str).str.lower().str.strip()

# Missing Value Imputation
num_cols = [c for c in df.select_dtypes(['int64', 'float64']).columns if c != 'id']
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())
for col in obj_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# 3. Categorical Derived Features (e.g. BMI category as string BEFORE categorical encoding)
def compute_bmi_category(bmi_val):
    if bmi_val < 18.5:
        return 'underweight'
    elif bmi_val < 25.0:
        return 'normal'
    elif bmi_val < 30.0:
        return 'overweight'
    else:
        return 'obese'

df['bmi_category'] = df['bmi'].apply(compute_bmi_category)

# 4. Target Encoding
le_target = LabelEncoder()
y_all = le_target.fit_transform(df[TARGET])
CLASS_NAMES = list(le_target.classes_)
print(f"  Target classes: {CLASS_NAMES}")
print(f"  Class distribution: {dict(zip(CLASS_NAMES, np.bincount(y_all)))}")

# 5. Categorical Feature Encoding (Store exact LabelEncoder for each column)
cat_encoders = {}
cat_cols = ['gender', 'diet_type', 'smoking_alcohol', 'sleep_quality', 
            'stress_level', 'physical_activity_level', 'bmi_category']

X_df = df.drop(columns=['id', TARGET], errors='ignore')

for col in cat_cols:
    if col in X_df.columns:
        le = LabelEncoder()
        # Sort classes deterministically
        unique_vals = sorted(X_df[col].astype(str).unique().tolist())
        le.fit(unique_vals)
        X_df[col] = le.transform(X_df[col].astype(str))
        cat_encoders[col] = le
        print(f"    Encoded '{col}': {dict(zip(le.classes_, range(len(le.classes_))))}")

# 6. Feature Engineering (Numerical Interaction Terms & Ratios)
def add_engineered_features(d):
    cols = d.columns.tolist()
    
    # Ratios & Efficiency
    if 'calorie_expenditure' in cols and 'exercise_duration' in cols:
        d['activity_efficiency'] = d['calorie_expenditure'] / (d['exercise_duration'] + 1.0)
    if 'step_count' in cols and 'calorie_expenditure' in cols:
        d['steps_per_calorie'] = d['step_count'] / (d['calorie_expenditure'] + 1.0)
    if 'heart_rate' in cols and 'sleep_duration' in cols:
        d['heart_rate_to_sleep'] = d['heart_rate'] / (d['sleep_duration'] + 0.5)
    if 'step_count' in cols and 'bmi' in cols:
        d['steps_per_bmi'] = d['step_count'] / (d['bmi'] + 0.1)
    if 'water_intake' in cols and 'exercise_duration' in cols:
        d['water_per_exercise'] = d['water_intake'] / (d['exercise_duration'] + 1.0)

    # Health score
    pos_feats = ['exercise_duration', 'sleep_duration', 'step_count', 'water_intake', 'calorie_expenditure']
    neg_feats = ['bmi', 'heart_rate']
    pos_score = sum(d[f] for f in pos_feats if f in cols)
    neg_score = sum(d[f] for f in neg_feats if f in cols)
    d['health_score'] = pos_score - neg_score

    # Sleep & Medical Flags
    if 'sleep_duration' in cols and 'sleep_quality' in cols:
        d['sleep_score'] = d['sleep_duration'] * d['sleep_quality']
        d['is_ideal_sleep'] = ((d['sleep_duration'] >= 7.0) & (d['sleep_duration'] <= 9.0)).astype(int)
        d['is_deprived_sleep'] = (d['sleep_duration'] < 6.0).astype(int)

    if 'bmi' in cols:
        d['is_normal_bmi'] = ((d['bmi'] >= 18.5) & (d['bmi'] <= 24.9)).astype(int)
        d['is_obese'] = (d['bmi'] >= 30.0).astype(int)

    if 'heart_rate' in cols:
        d['is_tachycardia'] = (d['heart_rate'] > 100.0).astype(int)
        d['is_bradycardia'] = (d['heart_rate'] < 60.0).astype(int)

    if 'step_count' in cols:
        d['is_active_steps'] = (d['step_count'] >= 8000).astype(int)

    if 'stress_level' in cols and 'sleep_duration' in cols:
        d['stress_sleep_ratio'] = d['stress_level'] / (d['sleep_duration'] + 1.0)

    # Baseline group diff features
    for mean_col in ['heart_rate_diff_from_group_mean', 'bmi_diff_from_group_mean', 
                      'step_count_diff_from_group_mean', 'calorie_expenditure_diff_from_group_mean']:
        if mean_col not in d.columns:
            d[mean_col] = 0.0

    return d

X_df = add_engineered_features(X_df)
FEATURE_NAMES = list(X_df.columns)
print(f"  Total Engineered Features: {len(FEATURE_NAMES)}")

# Enforce clean numerical float64 data types
X_df = X_df.apply(pd.to_numeric, errors='coerce').fillna(0.0).astype(np.float64)

# 7. Scaler
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_df), columns=FEATURE_NAMES)

# 8. Train / Validation Split (150K stratified sample for speed & accuracy)
SAMPLE_SIZE = min(150_000, len(X_scaled))
_, X_sub, _, y_sub = train_test_split(
    X_scaled, y_all,
    test_size=SAMPLE_SIZE/len(X_scaled),
    random_state=RS, stratify=y_all
)

X_tr, X_val, y_tr, y_val = train_test_split(
    X_sub, y_sub,
    test_size=0.20, random_state=RS, stratify=y_sub
)

print(f"  Training sample: {len(X_tr):,} rows | Validation sample: {len(X_val):,} rows")

# 9. Train Model (HistGradientBoostingClassifier)
print("  Training HistGradientBoostingClassifier...")
model = HistGradientBoostingClassifier(
    max_iter=300,
    learning_rate=0.05,
    max_depth=7,
    min_samples_leaf=20,
    class_weight='balanced',
    random_state=RS
)
model.fit(X_tr, y_tr)

# 10. Evaluate Model
val_preds = model.predict(X_val)
val_acc = accuracy_score(y_val, val_preds)
val_f1_w = f1_score(y_val, val_preds, average='weighted')
val_f1_m = f1_score(y_val, val_preds, average='macro')
conf_mat = confusion_matrix(y_val, val_preds).tolist()

print(f"  Validation Accuracy : {val_acc:.4f} ({val_acc*100:.2f}%)")
print(f"  Validation F1 (W)   : {val_f1_w:.4f}")
print(f"  Validation F1 (M)   : {val_f1_m:.4f}")
print("\n  Classification Report:\n", classification_report(y_val, val_preds, target_names=CLASS_NAMES))

# 11. Bundle Construction & Audit Metadata
training_timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

bundle = {
    'model': model,
    'model_name': 'HistGradientBoosting Classifier',
    'model_type': 'HistGradientBoostingClassifier',
    'scaler': scaler,
    'label_encoder': le_target,
    'categorical_encoders': cat_encoders,
    'feature_names': FEATURE_NAMES,
    'class_names': CLASS_NAMES,
    'class_mapping': {cls: int(idx) for idx, cls in enumerate(CLASS_NAMES)},
    'val_accuracy': round(float(val_acc), 4),
    'val_f1_weighted': round(float(val_f1_w), 4),
    'val_f1_macro': round(float(val_f1_m), 4),
    'confusion_matrix': conf_mat,
    'training_timestamp': training_timestamp,
    'environment_versions': {
        'python': sys.version.split()[0],
        'scikit_learn': sklearn.__version__,
        'numpy': np.__version__,
        'pandas': pd.__version__,
        'joblib': joblib.__version__
    },
    'project': 'CIS6005 Student Health Risk Prediction',
    'version': '2.0.0-deterministic'
}

bundle_path = MODELS / 'production_bundle.joblib'
joblib.dump(bundle, bundle_path)

# Calculate SHA256 Hash
sha256 = hashlib.sha256()
with open(bundle_path, 'rb') as f:
    while chunk := f.read(8192):
        sha256.update(chunk)
file_hash = sha256.hexdigest()

bundle['sha256'] = file_hash
# Re-dump with hash metadata included
joblib.dump(bundle, bundle_path)

print(f"\n  [OK] SAVED PRODUCTION BUNDLE:")
print(f"     Path   : {bundle_path}")
print(f"     SHA256 : {file_hash}")
print(f"     Size   : {bundle_path.stat().st_size / (1024*1024):.2f} MB")
print("=" * 65)
