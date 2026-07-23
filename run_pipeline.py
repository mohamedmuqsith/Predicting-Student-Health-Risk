"""
CIS6005 — Student Health Risk Prediction
Master Pipeline Runner — FAST VERSION (completes in ~5 mins)
Uses 100K stratified sample for training + full dataset for final predictions
Run: python run_pipeline.py
"""
import warnings
warnings.filterwarnings('ignore')
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import pandas as pd
import joblib, time, datetime
from pathlib import Path

from sklearn.preprocessing   import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, RandomizedSearchCV
from sklearn.linear_model    import LogisticRegression
from sklearn.neighbors       import KNeighborsClassifier
from sklearn.tree            import DecisionTreeClassifier
from sklearn.ensemble        import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics         import accuracy_score, f1_score, classification_report

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
    print(f"\n{'='*60}\n  {msg}\n{'='*60}", flush=True)

# ── PHASE 1: LOAD ─────────────────────────────────────────
banner("PHASE 1 — Loading Raw Data")
train_raw = pd.read_csv(RAW / 'train.csv')
test_raw  = pd.read_csv(RAW / 'test.csv')
print(f"  Train: {train_raw.shape}  |  Test: {test_raw.shape}")
print(f"  Target: {sorted(train_raw['health_condition'].dropna().unique())}")

# ── PHASE 2: CLEAN ────────────────────────────────────────
banner("PHASE 2 — Cleaning Data")
train_df = train_raw.copy()
test_df  = test_raw.copy()

# String normalisation (all object cols except target)
obj_cols = [c for c in train_df.select_dtypes('object').columns if c != 'health_condition']
for col in obj_cols:
    train_df[col] = train_df[col].astype(str).str.lower().str.strip()
    if col in test_df.columns:
        test_df[col] = test_df[col].astype(str).str.lower().str.strip()
train_df['health_condition'] = train_df['health_condition'].astype(str).str.lower().str.strip()

# Impute missing values
num_cols = [c for c in train_df.select_dtypes(['int64','float64']).columns if c != 'id']
cat_cols = [c for c in train_df.select_dtypes('object').columns if c != 'health_condition']

for col in num_cols:
    fill = train_df[col].median()
    train_df[col] = train_df[col].fillna(fill)
    if col in test_df.columns:
        test_df[col] = test_df[col].fillna(fill)

for col in cat_cols:
    fill = train_df[col].mode()[0]
    train_df[col] = train_df[col].fillna(fill)
    if col in test_df.columns:
        test_df[col] = test_df[col].fillna(fill)

print(f"  Numeric cols   : {num_cols}")
print(f"  Categorical cols: {cat_cols}")
print(f"  Missing values : {train_df.isnull().sum().sum()}")

# ── PHASE 3: FEATURE ENGINEERING ─────────────────────────
banner("PHASE 3 — Feature Engineering")

def engineer(df):
    d = df.copy()
    nc = d.select_dtypes(['int64','float64']).columns.tolist()

    def mm(c):
        mn, mx = d[c].min(), d[c].max()
        return (d[c] - mn) / (mx - mn + 1e-9)

    pos = [f for f in ['exercise_duration','sleep_duration','step_count','water_intake','calorie_expenditure'] if f in nc]
    neg = [f for f in ['bmi','heart_rate'] if f in nc]
    d['health_score']        = sum(mm(f) for f in pos) - sum(mm(f) for f in neg)

    if 'calorie_expenditure' in nc and 'exercise_duration' in nc:
        d['activity_efficiency'] = d['calorie_expenditure'] / (d['exercise_duration'] + 1)

    if 'step_count' in nc and 'calorie_expenditure' in nc:
        d['steps_per_calorie'] = d['step_count'] / (d['calorie_expenditure'] + 1)

    if 'sleep_duration' in nc:
        d['sleep_score'] = d['sleep_duration']

    if 'bmi' in nc:
        d['bmi_category'] = pd.cut(d['bmi'],
            bins=[-np.inf, 18.5, 25.0, 30.0, np.inf],
            labels=[0, 1, 2, 3]).astype(float)
    return d

train_df = engineer(train_df)
test_df  = engineer(test_df)
new_feats = [c for c in train_df.columns if c not in train_raw.columns]
print(f"  New features: {new_feats}")

# ── PHASE 4: ENCODE + SCALE ───────────────────────────────
banner("PHASE 4 — Encoding & Scaling")

TARGET     = 'health_condition'
X_all      = train_df.drop(columns=['id', TARGET], errors='ignore')
y_all      = train_df[TARGET]
test_ids   = test_df['id'] if 'id' in test_df.columns else pd.Series(range(len(test_df)))
X_test_raw = test_df.drop(columns=['id', TARGET], errors='ignore')

le_target   = LabelEncoder()
y_enc       = le_target.fit_transform(y_all)
CLASS_NAMES = list(le_target.classes_)
print(f"  Classes: {dict(zip(CLASS_NAMES, le_target.transform(CLASS_NAMES).tolist()))}")

cat_encoders = {}
feat_cat_cols = X_all.select_dtypes('object').columns.tolist()
for col in feat_cat_cols:
    le = LabelEncoder()
    X_all[col] = le.fit_transform(X_all[col].astype(str))
    tc = X_test_raw[col].astype(str).apply(lambda v: v if v in set(le.classes_) else le.classes_[0])
    X_test_raw[col] = le.transform(tc)
    cat_encoders[col] = le
    print(f"  Encoded: {col} -> {list(le.classes_)}")

X_all      = X_all.apply(pd.to_numeric, errors='coerce').fillna(0)
X_test_raw = X_test_raw.apply(pd.to_numeric, errors='coerce').fillna(0)
X_test_raw = X_test_raw.reindex(columns=X_all.columns, fill_value=0)

scaler      = StandardScaler()
X_all_sc    = pd.DataFrame(scaler.fit_transform(X_all),      columns=X_all.columns)
X_test_sc   = pd.DataFrame(scaler.transform(X_test_raw),     columns=X_all.columns)
feat_names  = list(X_all.columns)
print(f"  Total features: {len(feat_names)}")

# ── STRATIFIED 100K SAMPLE for training ───────────────────
SAMPLE_SIZE = 100_000
print(f"\n  Full dataset: {len(X_all_sc):,} rows")
print(f"  Using stratified {SAMPLE_SIZE:,} sample for training (faster, accurate)")

_, X_s, _, y_s = train_test_split(
    X_all_sc, y_enc,
    test_size=SAMPLE_SIZE/len(X_all_sc),
    random_state=RS, stratify=y_enc
)

X_tr, X_val, y_tr, y_val = train_test_split(
    X_s, y_s,
    test_size=0.20, random_state=RS, stratify=y_s
)
print(f"  Train: {X_tr.shape}  |  Val: {X_val.shape}")

# Save for notebooks
np.save(PROC / 'X_train.npy', X_tr.values)
np.save(PROC / 'X_val.npy',   X_val.values)
np.save(PROC / 'y_train.npy', y_tr)
np.save(PROC / 'y_val.npy',   y_val)
np.save(PROC / 'X_test.npy',  X_test_sc.values)
joblib.dump(le_target,    MODELS / 'label_encoder.joblib')
joblib.dump(scaler,       MODELS / 'scaler.joblib')
joblib.dump(cat_encoders, MODELS / 'categorical_encoders.joblib')
joblib.dump(feat_names,   MODELS / 'feature_names.joblib')
print("  Preprocessing artifacts saved.")

# ── PHASE 5: TRAIN ALL MODELS ─────────────────────────────
banner("PHASE 5 — Training 5 Models")

models_cfg = [
    ('Logistic Regression',  LogisticRegression(max_iter=500, class_weight='balanced',
                                                 solver='lbfgs', random_state=RS)),
    ('K-Nearest Neighbours', KNeighborsClassifier(n_neighbors=7, weights='distance', n_jobs=-1)),
    ('Decision Tree',        DecisionTreeClassifier(max_depth=12, min_samples_split=20,
                                                    class_weight='balanced', random_state=RS)),
    ('Random Forest',        RandomForestClassifier(n_estimators=100, max_depth=15,
                                                    min_samples_split=10, class_weight='balanced',
                                                    n_jobs=-1, random_state=RS)),
    ('Gradient Boosting',    GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
                                                         max_depth=5, random_state=RS)),
]
fname_map = {
    'Logistic Regression' : 'model_logistic_regression.joblib',
    'K-Nearest Neighbours': 'model_knn.joblib',
    'Decision Tree'       : 'model_decision_tree.joblib',
    'Random Forest'       : 'model_random_forest.joblib',
    'Gradient Boosting'   : 'model_gradient_boosting.joblib',
}

results = {}
print(f"  {'Model':<25} {'Val Acc':>9} {'Val F1-W':>9} {'Time':>7}")
print(f"  {'-'*55}")

for name, mdl in models_cfg:
    t0 = time.time()
    mdl.fit(X_tr.values, y_tr)
    yp  = mdl.predict(X_val.values)
    acc = accuracy_score(y_val, yp)
    f1  = f1_score(y_val, yp, average='weighted')
    t   = time.time() - t0
    results[name] = {'model': mdl, 'acc': acc, 'f1': f1}
    joblib.dump(mdl, MODELS / fname_map[name])
    print(f"  {name:<25} {acc:>9.4f} {f1:>9.4f} {t:>6.1f}s", flush=True)

# ── PHASE 6: CROSS-VALIDATION (top 2 only) ────────────────
banner("PHASE 6 — 5-Fold Cross-Validation")

cv  = StratifiedKFold(n_splits=5, shuffle=True, random_state=RS)
Xsv = X_s.values if hasattr(X_s, 'values') else X_s
top2 = sorted(results, key=lambda k: results[k]['f1'], reverse=True)[:2]
cv_res = {}

print(f"  Running CV on top 2: {top2}")
print(f"  {'Model':<25} {'CV Mean':>9} {'CV Std':>9}")
print(f"  {'-'*45}")

for name in top2:
    sc = cross_val_score(results[name]['model'], Xsv, y_s,
                         cv=cv, scoring='f1_weighted', n_jobs=-1)
    cv_res[name] = sc
    print(f"  {name:<25} {sc.mean():>9.4f} {sc.std():>9.4f}", flush=True)

for name in results:
    if name not in cv_res:
        cv_res[name] = np.array([results[name]['f1']] * 5)

# ── PHASE 7: HYPERPARAMETER TUNING ────────────────────────
banner("PHASE 7 — Hyperparameter Tuning (RandomizedSearchCV, 15 iter)")

rf_params = {
    'n_estimators'     : [100, 150, 200],
    'max_depth'        : [10, 12, 15, 20],
    'min_samples_split': [5, 10, 15],
    'min_samples_leaf' : [1, 2, 4],
}

rscv = RandomizedSearchCV(
    RandomForestClassifier(class_weight='balanced', n_jobs=-1, random_state=RS),
    rf_params, n_iter=15, cv=cv, scoring='f1_weighted',
    n_jobs=-1, verbose=0, refit=True, random_state=RS
)
rscv.fit(Xsv, y_s)

tuned  = rscv.best_estimator_
ytp    = tuned.predict(X_val.values)
tacc   = accuracy_score(y_val, ytp)
tf1    = f1_score(y_val, ytp, average='weighted')

print(f"  Best params : {rscv.best_params_}")
print(f"  Best CV F1  : {rscv.best_score_:.4f}")
print(f"  Val Accuracy: {tacc:.4f}  |  Val F1-W: {tf1:.4f}")

joblib.dump(tuned, MODELS / 'model_tuned_best.joblib')
cv_res['Tuned RF'] = np.array([rscv.best_score_] * 5)
results['Tuned RF'] = {'model': tuned, 'acc': tacc, 'f1': tf1}

# ── PHASE 8: CHAMPION SELECTION ───────────────────────────
banner("PHASE 8 — Champion Model Selection")

best_name  = max(cv_res, key=lambda k: cv_res[k].mean())
best_model = results[best_name]['model']
ycp        = best_model.predict(X_val.values)
final_acc  = accuracy_score(y_val, ycp)
final_f1w  = f1_score(y_val, ycp, average='weighted')
final_f1m  = f1_score(y_val, ycp, average='macro')

print(f"  Champion     : {best_name}")
print(f"  Val Accuracy : {final_acc:.4f}  ({final_acc*100:.2f}%)")
print(f"  F1 Weighted  : {final_f1w:.4f}")
print(f"  F1 Macro     : {final_f1m:.4f}")
print()
print(classification_report(y_val, ycp, target_names=CLASS_NAMES))

# Save comparison table
comp = pd.DataFrame([{
    'Model'      : n,
    'Val Acc'    : round(results[n]['acc'], 4),
    'Val F1-W'   : round(results[n]['f1'],  4),
    'CV F1 Mean' : round(cv_res[n].mean(),  4),
    'CV F1 Std'  : round(cv_res[n].std(),   4),
} for n in results]).sort_values('CV F1 Mean', ascending=False)
comp.to_csv(PROC / 'final_comparison.csv', index=False)
print(comp.to_string(index=False))

# ── PHASE 9: KAGGLE SUBMISSION ────────────────────────────
banner("PHASE 9 — Kaggle Submission")

y_test_pred = le_target.inverse_transform(
    best_model.predict(np.load(PROC / 'X_test.npy'))
)
sub_df = pd.DataFrame({'id': test_ids.values, 'health_condition': y_test_pred})
ts     = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
fname  = f'submission_{best_name.replace(" ","_")}_{ts}.csv'
sub_df.to_csv(SUBMIT / fname, index=False)

u, c = np.unique(y_test_pred, return_counts=True)
for cls, cnt in zip(u, c):
    print(f"  {cls:<15}: {cnt:>7,}  ({cnt/len(y_test_pred)*100:.1f}%)")
print(f"  Total         : {len(sub_df):>7,}")
print(f"  Saved: {fname}")

# ── PHASE 10: PRODUCTION BUNDLE ───────────────────────────
banner("PHASE 10 — Production Bundle")

bundle = {
    'model'               : best_model,
    'model_name'          : best_name,
    'model_type'          : type(best_model).__name__,
    'scaler'              : scaler,
    'label_encoder'       : le_target,
    'categorical_encoders': cat_encoders,
    'feature_names'       : feat_names,
    'class_names'         : CLASS_NAMES,
    'class_mapping'       : {k: int(v) for k, v in zip(
                                CLASS_NAMES, le_target.transform(CLASS_NAMES).tolist())},
    'val_accuracy'        : round(final_acc, 4),
    'val_f1_weighted'     : round(final_f1w, 4),
    'val_f1_macro'        : round(final_f1m, 4),
    'project'             : 'CIS6005 Student Health Risk Prediction',
    'competition'         : 'Kaggle Playground Series S6E7',
    'created_at'          : datetime.datetime.now().isoformat(),
}

joblib.dump(bundle,     MODELS / 'production_bundle.joblib')
joblib.dump(best_model, MODELS / 'best_model_final.joblib')
print(f"  models/production_bundle.joblib  [OK]")
print(f"  models/best_model_final.joblib   [OK]")

# ── FINAL SUMMARY ─────────────────────────────────────────
banner("PIPELINE COMPLETE")
print(f"  Champion     : {best_name}")
print(f"  Val Accuracy : {final_acc*100:.2f}%")
print(f"  F1 Weighted  : {final_f1w:.4f}")
print(f"  Submission   : {fname}")
print()
print("  All models trained and saved  [OK]")
print("  Production bundle ready       [OK]")
print("  Kaggle submission created     [OK]")
print()
print("  Launching Streamlit App...")
print("  URL: http://localhost:8501")
print("="*60, flush=True)

time.sleep(2)
import subprocess, sys as _sys
subprocess.run([
    _sys.executable, "-m", "streamlit", "run",
    str(ROOT / "streamlit_app" / "app.py"),
    "--server.port=8501",
    "--browser.gatherUsageStats=false"
])
