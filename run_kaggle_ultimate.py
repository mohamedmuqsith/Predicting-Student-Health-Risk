"""
CIS6005 / Kaggle ULTIMATE Fast Pipeline — Student Health Risk Prediction
Strategy: 5-Fold OOF Stacking (LightGBM + XGBoost + CatBoost + HistGB)
          + Logistic Regression Meta-Learner
Target: 0.97+ Public Score
ETA: ~15-20 minutes on full 690k dataset

Run: python run_kaggle_ultimate.py
"""
import warnings
warnings.filterwarnings('ignore')

import sys, io, time, datetime
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, classification_report, balanced_accuracy_score
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_sample_weight

import lightgbm as lgb
import xgboost as xgb

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

ROOT   = Path(__file__).parent
RAW    = ROOT / 'data' / 'raw'
MODELS = ROOT / 'models'
SUBMIT = ROOT / 'data' / 'submissions'
for d in [MODELS, SUBMIT]:
    d.mkdir(parents=True, exist_ok=True)

RS    = 42
FOLDS = 5
np.random.seed(RS)

def banner(msg):
    print(f"\n{'='*70}\n  {msg}\n{'='*70}", flush=True)

# ── 1. LOAD ────────────────────────────────────────────────────
banner("PHASE 1 — Loading Full Dataset")
train_raw = pd.read_csv(RAW / 'train.csv')
test_raw  = pd.read_csv(RAW / 'test.csv')
print(f"  Train: {train_raw.shape} | Test: {test_raw.shape}", flush=True)

# ── 2. FEATURE ENGINEERING ────────────────────────────────────
banner("PHASE 2 — Ultimate Feature Engineering")

def create_features(df_input):
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

    stress_map   = {'low':1,'medium':2,'high':3}
    sleep_q_map  = {'poor':1,'fair':2,'good':3,'excellent':4}
    activity_map = {'sedentary':1,'moderate':2,'active':3,'very active':4}
    smoke_map    = {'no':0,'yes':1}

    df['stress_num']   = df['stress_level'].map(stress_map).fillna(2)
    df['sleep_q_num']  = df['sleep_quality'].map(sleep_q_map).fillna(2)
    df['activity_num'] = df['physical_activity_level'].map(activity_map).fillna(2)
    df['smoke_num']    = df['smoking_alcohol'].map(smoke_map).fillna(0)

    # Ratios
    df['activity_efficiency'] = df['calorie_expenditure'] / (df['exercise_duration'] + 1.0)
    df['steps_per_calorie']   = df['step_count'] / (df['calorie_expenditure'] + 1.0)
    df['heart_rate_to_sleep'] = df['heart_rate'] / (df['sleep_duration'] + 0.5)
    df['steps_per_bmi']       = df['step_count'] / (df['bmi'] + 0.1)
    df['water_per_exercise']  = df['water_intake'] / (df['exercise_duration'] + 1.0)
    df['calorie_per_step']    = df['calorie_expenditure'] / (df['step_count'] + 1.0)
    df['sleep_per_stress']    = df['sleep_duration'] / (df['stress_num'] + 0.1)
    df['bmi_per_activity']    = df['bmi'] / (df['activity_num'] + 0.1)

    # Non-linear
    df['bmi_sq']       = df['bmi'] ** 2
    df['hr_sq']        = df['heart_rate'] ** 2
    df['step_log']     = np.log1p(df['step_count'])
    df['calorie_log']  = np.log1p(df['calorie_expenditure'])
    df['exercise_log'] = np.log1p(df['exercise_duration'])
    df['bmi_sqrt']     = np.sqrt(df['bmi'].clip(0))

    # Composite indices
    df['sleep_quality_index'] = df['sleep_duration'] * df['sleep_q_num']
    df['stress_bmi_load']     = df['bmi'] * df['stress_num']
    df['cardio_strain']       = (df['heart_rate'] * df['stress_num']) / (df['sleep_duration'] + 0.5)
    df['health_risk_score']   = (
        df['bmi'] * 0.4 + df['heart_rate'] * 0.3 + df['stress_num'] * 5.0 +
        df['smoke_num'] * 10.0 - df['sleep_quality_index'] * 2.0 - df['step_count'] / 1000.0
    )
    df['wellness_score']  = (
        df['sleep_q_num'] * 3.0 + df['activity_num'] * 2.0 +
        df['step_log'] - df['stress_num'] * 2.0 - df['smoke_num'] * 5.0
    )
    df['recovery_index'] = df['sleep_duration'] * df['sleep_q_num'] / (df['stress_num'] + 0.1)
    df['cardio_fitness'] = df['step_count'] / (df['heart_rate'] + 1.0) * df['activity_num']

    # Medical binary indicators
    df['is_ideal_sleep']    = ((df['sleep_duration'] >= 7) & (df['sleep_duration'] <= 9)).astype(int)
    df['is_deprived_sleep'] = (df['sleep_duration'] < 6).astype(int)
    df['is_normal_bmi']     = ((df['bmi'] >= 18.5) & (df['bmi'] <= 24.9)).astype(int)
    df['is_overweight']     = ((df['bmi'] >= 25) & (df['bmi'] < 30)).astype(int)
    df['is_obese']          = (df['bmi'] >= 30).astype(int)
    df['is_underweight']    = (df['bmi'] < 18.5).astype(int)
    df['is_tachycardia']    = (df['heart_rate'] > 100).astype(int)
    df['is_bradycardia']    = (df['heart_rate'] < 60).astype(int)
    df['is_active_steps']   = (df['step_count'] >= 8000).astype(int)
    df['is_very_active']    = (df['step_count'] >= 12000).astype(int)
    df['is_high_stress']    = (df['stress_num'] == 3).astype(int)
    df['is_smoker']         = (df['smoke_num'] == 1).astype(int)
    df['is_excellent_sleep']= (df['sleep_q_num'] == 4).astype(int)
    df['is_poor_sleep']     = (df['sleep_q_num'] == 1).astype(int)

    # Triple interactions
    df['stress_smoke_bmi']    = df['stress_num'] * df['smoke_num'] * df['bmi']
    df['active_sleep_stress'] = df['activity_num'] * df['sleep_q_num'] / (df['stress_num'] + 0.1)
    df['risk_activity_combo'] = df['health_risk_score'] / (df['activity_num'] + 0.1)

    return df

df_full = pd.concat([
    train_raw.assign(is_train=1),
    test_raw.assign(is_train=0, health_condition=np.nan)
], ignore_index=True)
df_full = create_features(df_full)

# Group aggregation features
group_combos = [
    ['gender', 'physical_activity_level'],
    ['gender', 'diet_type'],
    ['stress_level', 'sleep_quality'],
    ['gender', 'smoking_alcohol'],
    ['physical_activity_level', 'diet_type'],
]
agg_cols = ['heart_rate','bmi','step_count','calorie_expenditure',
            'sleep_duration','health_risk_score','wellness_score']

for g_cols in group_combos:
    prefix = "_".join(g_cols)
    for ac in agg_cols:
        if ac not in df_full.columns:
            continue
        gm = df_full.groupby(g_cols)[ac].transform('mean')
        gs = df_full.groupby(g_cols)[ac].transform('std').fillna(1.0)
        df_full[f'{ac}_diff_{prefix}']   = df_full[ac] - gm
        df_full[f'{ac}_zscore_{prefix}'] = (df_full[ac] - gm) / (gs + 1e-6)

train_proc = df_full[df_full['is_train'] == 1].drop(columns=['is_train'])
test_proc  = df_full[df_full['is_train'] == 0].drop(columns=['is_train'])
print(f"  Total Features: {train_proc.shape[1] - 2}", flush=True)

# ── 3. ENCODING ────────────────────────────────────────────────
banner("PHASE 3 — Encoding & Scaling")
TARGET = 'health_condition'
le_target = LabelEncoder()
y_all = le_target.fit_transform(train_proc[TARGET])
CLASS_NAMES = list(le_target.classes_)
N_CLASSES   = len(CLASS_NAMES)
print(f"  Classes: {CLASS_NAMES}", flush=True)

X_all  = train_proc.drop(columns=['id', TARGET], errors='ignore')
test_ids = test_proc['id'] if 'id' in test_proc.columns else pd.Series(range(len(test_proc)))
X_test = test_proc.drop(columns=['id', TARGET], errors='ignore')

cat_cols  = X_all.select_dtypes('object').columns.tolist()
encoders  = {}
for c in cat_cols:
    le = LabelEncoder()
    le.fit(pd.concat([X_all[c].astype(str), X_test[c].astype(str)]))
    X_all[c]  = le.transform(X_all[c].astype(str))
    X_test[c] = le.transform(X_test[c].astype(str))
    encoders[c] = le

scaler   = StandardScaler()
X_all_s  = pd.DataFrame(scaler.fit_transform(X_all),  columns=X_all.columns)
X_test_s = pd.DataFrame(scaler.transform(X_test),     columns=X_all.columns)

# ── 4. 5-FOLD OOF STACKING ────────────────────────────────────
N_MODELS = 4 + (1 if HAS_CATBOOST else 0)
banner(f"PHASE 4 — {FOLDS}-Fold OOF Stacking ({N_MODELS} Models)")

skf          = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=RS)
oof_preds    = np.zeros((len(X_all_s),  N_CLASSES, N_MODELS))
test_preds   = np.zeros((len(X_test_s), N_CLASSES, N_MODELS))
fold_scores  = []

for fold, (tr_idx, val_idx) in enumerate(skf.split(X_all_s, y_all)):
    t_fold = time.time()
    print(f"\n  ── FOLD {fold+1}/{FOLDS} ({len(tr_idx):,} train | {len(val_idx):,} val) ──", flush=True)
    Xtr, Xvl = X_all_s.iloc[tr_idx], X_all_s.iloc[val_idx]
    ytr, yvl = y_all[tr_idx], y_all[val_idx]
    sw = compute_sample_weight('balanced', ytr)

    # Model 0: LightGBM (1st place configuration)
    t0 = time.time()
    m_lgb = lgb.LGBMClassifier(
        n_estimators=450, learning_rate=0.03, num_leaves=63, max_depth=7,
        min_child_samples=30, class_weight='balanced',
        subsample=0.85, colsample_bytree=0.85,
        random_state=RS+fold, n_jobs=-1, verbose=-1
    )
    m_lgb.fit(Xtr, ytr)
    oof_preds[val_idx, :, 0]  = m_lgb.predict_proba(Xvl)
    test_preds[:, :, 0]       += m_lgb.predict_proba(X_test_s) / FOLDS
    print(f"    [1/{N_MODELS}] LightGBM   {time.time()-t0:.0f}s", flush=True)

    # Model 1: XGBoost (1st place configuration)
    t0 = time.time()
    m_xgb = xgb.XGBClassifier(
        n_estimators=400, learning_rate=0.03, max_depth=6,
        subsample=0.85, colsample_bytree=0.85,
        random_state=RS+fold, n_jobs=-1, verbosity=0
    )
    m_xgb.fit(Xtr, ytr, sample_weight=sw)
    oof_preds[val_idx, :, 1]  = m_xgb.predict_proba(Xvl)
    test_preds[:, :, 1]       += m_xgb.predict_proba(X_test_s) / FOLDS
    print(f"    [2/{N_MODELS}] XGBoost    {time.time()-t0:.0f}s", flush=True)

    # Model 2: HistGradientBoosting (1st place configuration)
    t0 = time.time()
    m_hgb = HistGradientBoostingClassifier(
        max_iter=350, learning_rate=0.03, max_leaf_nodes=63,
        class_weight='balanced', random_state=RS+fold
    )
    m_hgb.fit(Xtr, ytr)
    oof_preds[val_idx, :, 2]  = m_hgb.predict_proba(Xvl)
    test_preds[:, :, 2]       += m_hgb.predict_proba(X_test_s) / FOLDS
    print(f"    [3/{N_MODELS}] HistGB     {time.time()-t0:.0f}s", flush=True)

    # Model 3: ExtraTrees (replaced LGBM2 for model diversity)
    t0 = time.time()
    m_et = ExtraTreesClassifier(
        n_estimators=150, max_depth=16, min_samples_split=10,
        class_weight='balanced', n_jobs=-1, random_state=RS+fold
    )
    m_et.fit(Xtr, ytr)
    oof_preds[val_idx, :, 3]  = m_et.predict_proba(Xvl)
    test_preds[:, :, 3]       += m_et.predict_proba(X_test_s) / FOLDS
    print(f"    [4/{N_MODELS}] ExtraTrees {time.time()-t0:.0f}s", flush=True)

    # Model 4: CatBoost (if available)
    if HAS_CATBOOST:
        t0 = time.time()
        m_cat = CatBoostClassifier(
            iterations=300, learning_rate=0.05, depth=7,
            loss_function='MultiClass', auto_class_weights='Balanced',
            random_seed=RS+fold, task_type='CPU', verbose=0, thread_count=-1
        )
        m_cat.fit(Xtr, ytr)
        oof_preds[val_idx, :, 4]  = m_cat.predict_proba(Xvl)
        test_preds[:, :, 4]       += m_cat.predict_proba(X_test_s) / FOLDS
        print(f"    [5/{N_MODELS}] CatBoost   {time.time()-t0:.0f}s", flush=True)

    # Fold score
    avg_oof = np.mean(oof_preds[val_idx], axis=2)
    bal_acc = balanced_accuracy_score(yvl, np.argmax(avg_oof, axis=1))
    fold_scores.append(bal_acc)
    print(f"    FOLD {fold+1} OOF BalAcc={bal_acc:.4f} | total={time.time()-t_fold:.0f}s", flush=True)

print(f"\n  ✅ Mean OOF Balanced Accuracy: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}", flush=True)

# Save OOF arrays for crash recovery
np.save(MODELS / 'oof_preds.npy',  oof_preds)
np.save(MODELS / 'test_preds.npy', test_preds)
np.save(MODELS / 'y_all.npy',      y_all)
print("  Saved OOF arrays to models/ [OK]", flush=True)

# ── 5. META-LEARNER & OPTIMIZATION ────────────────────────────
banner("PHASE 5 — Meta-Learner & Blend Optimization for Balanced Accuracy")

# Evaluate multiple strategies to find the absolute best
# 1. Simple Average of all models
simple_all_preds = np.argmax(np.mean(oof_preds, axis=2), axis=1)
simple_all_score = balanced_accuracy_score(y_all, simple_all_preds)
print(f"  1. Simple Average (All {N_MODELS} models) BalAcc: {simple_all_score:.6f}")

# 2. Balanced Logistic Regression Meta-Learner on all models
oof_meta  = oof_preds.reshape(len(oof_preds), N_CLASSES * N_MODELS)
test_meta = test_preds.reshape(len(test_preds), N_CLASSES * N_MODELS)

meta_lr = LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs', class_weight='balanced', random_state=RS)
meta_lr.fit(oof_meta, y_all)
meta_lr_preds = meta_lr.predict(oof_meta)
meta_lr_score = balanced_accuracy_score(y_all, meta_lr_preds)
print(f"  2. Balanced Meta-LR Stack (All models) BalAcc   : {meta_lr_score:.6f}")

# 3. Optimized Weight Blending
def optimize_weights_fn(probs_to_blend):
    from scipy.optimize import minimize
    def loss_func(weights):
        w = np.abs(weights)
        if w.sum() == 0:
            return 1.0
        w = w / w.sum()
        weighted_probs = np.zeros((len(y_all), N_CLASSES))
        for i in range(probs_to_blend.shape[2]):
            weighted_probs += probs_to_blend[:, :, i] * w[i]
        preds = np.argmax(weighted_probs, axis=1)
        return -balanced_accuracy_score(y_all, preds)

    init_weights = [1.0 / probs_to_blend.shape[2]] * probs_to_blend.shape[2]
    res = minimize(loss_func, init_weights, method='Nelder-Mead', options={'maxiter': 200})
    best_w = np.abs(res.x)
    return best_w / best_w.sum(), -res.fun

w_opt, opt_score = optimize_weights_fn(oof_preds)
print(f"  3. Optimized Weights (All models) BalAcc       : {opt_score:.6f}  (Weights: {w_opt})")

# Selection dict
strategies = {
    "Simple Average Blend": (simple_all_score, lambda: le_target.inverse_transform(np.argmax(np.mean(test_preds, axis=2), axis=1))),
    "Balanced Meta-Learner Stack": (meta_lr_score, lambda: le_target.inverse_transform(meta_lr.predict(test_meta))),
    "Optimized Weights Blend": (opt_score, lambda: le_target.inverse_transform(np.argmax(sum(test_preds[:, :, i] * w_opt[i] for i in range(N_MODELS)), axis=1)))
}

strategy = max(strategies, key=lambda k: strategies[k][0])
best_score, get_labels_fn = strategies[strategy]
final_labels = get_labels_fn()

print(f"\n  ✅ Selected Strategy: {strategy} (OOF Balanced Accuracy: {best_score:.6f})", flush=True)

# ── 6. SUBMISSION ─────────────────────────────────────────────
banner("PHASE 6 — Generating Ultimate Submission")

sub_df = pd.DataFrame({'id': test_ids.values, 'health_condition': final_labels})
ts   = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
name = f'submission_Kaggle_ULTIMATE_OOF_Stack_{ts}.csv'
path = SUBMIT / name
sub_df.to_csv(path, index=False)

print("  Class Breakdown:")
u, c = np.unique(final_labels, return_counts=True)
for cls, cnt in zip(u, c):
    print(f"    {cls:<15}: {cnt:>7,}  ({cnt/len(final_labels)*100:.1f}%)")

print(f"\n  🏆 SUBMISSION READY!")
print(f"     Strategy  : {strategy}")
print(f"     OOF BalAcc: {best_score:.6f}")
print(f"     File      : {name}")
print(f"     Path      : {path}")

joblib.dump({
    'meta_lr': meta_lr, 'scaler': scaler, 'label_encoder': le_target,
    'cat_encoders': encoders, 'feature_names': list(X_all.columns),
    'class_names': CLASS_NAMES, 'oof_score': round(best_score, 6), 'strategy': strategy,
    'opt_weights': w_opt
}, MODELS / 'production_bundle.joblib')
print("  Saved production_bundle.joblib [OK]")
banner("ULTIMATE PIPELINE COMPLETE")































