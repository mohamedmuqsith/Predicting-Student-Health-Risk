"""
Quick Meta-Learner Fix — Runs Phase 5 & 6 only using saved OOF numpy arrays.
Optimizes specifically for BALANCED ACCURACY (the Kaggle S6E7 evaluation metric).
Run: python run_meta_only.py
"""
import warnings
warnings.filterwarnings('ignore')
import sys, io, datetime
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, classification_report
from scipy.optimize import minimize

ROOT   = Path(__file__).parent
MODELS = ROOT / 'models'
SUBMIT = ROOT / 'data' / 'submissions'

def banner(msg):
    print(f"\n{'='*70}\n  {msg}\n{'='*70}", flush=True)

# Load saved arrays
print("Loading saved OOF arrays...", flush=True)
oof_preds  = np.load(MODELS / 'oof_preds.npy')
test_preds = np.load(MODELS / 'test_preds.npy')
y_all      = np.load(MODELS / 'y_all.npy')
test_ids   = pd.read_csv(ROOT / 'data' / 'raw' / 'test.csv')['id']

bundle     = joblib.load(MODELS / 'production_bundle.joblib')
le_target  = bundle['label_encoder']
CLASS_NAMES= bundle['class_names']
N_CLASSES  = len(CLASS_NAMES)
N_MODELS   = oof_preds.shape[2]

banner("PHASE 5 — Meta-Learner & Blend Optimization for Balanced Accuracy")

# 1. Simple Average of all 5 models
simple_all_preds = np.argmax(np.mean(oof_preds, axis=2), axis=1)
simple_all_score = balanced_accuracy_score(y_all, simple_all_preds)
print(f"1. Simple Average (All 5 models) Balanced Accuracy: {simple_all_score:.6f}")

# 2. Simple Average of models 0-3 (excluding unbalanced CatBoost)
if N_MODELS > 4:
    simple_4_preds = np.argmax(np.mean(oof_preds[:, :, :4], axis=2), axis=1)
    simple_4_score = balanced_accuracy_score(y_all, simple_4_preds)
    print(f"2. Simple Average (Models 0-3) Balanced Accuracy  : {simple_4_score:.6f}")
else:
    simple_4_score = 0.0

# 3. Logistic Regression Meta-Learner (Balanced) on all models
oof_meta  = oof_preds.reshape(len(oof_preds), N_CLASSES * N_MODELS)
test_meta = test_preds.reshape(len(test_preds), N_CLASSES * N_MODELS)

meta_lr = LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs', class_weight='balanced', random_state=42)
meta_lr.fit(oof_meta, y_all)
meta_lr_preds = meta_lr.predict(oof_meta)
meta_lr_score = balanced_accuracy_score(y_all, meta_lr_preds)
print(f"3. Balanced Meta-LR Stack (All models) Bal Accuracy  : {meta_lr_score:.6f}")

# 4. Logistic Regression Meta-Learner (Balanced) on models 0-3
if N_MODELS > 4:
    oof_meta_4 = oof_preds[:, :, :4].reshape(len(oof_preds), N_CLASSES * 4)
    test_meta_4 = test_preds[:, :, :4].reshape(len(test_preds), N_CLASSES * 4)
    meta_lr_4 = LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs', class_weight='balanced', random_state=42)
    meta_lr_4.fit(oof_meta_4, y_all)
    meta_lr_4_preds = meta_lr_4.predict(oof_meta_4)
    meta_lr_4_score = balanced_accuracy_score(y_all, meta_lr_4_preds)
    print(f"4. Balanced Meta-LR Stack (Models 0-3) Bal Accuracy : {meta_lr_4_score:.6f}")
else:
    meta_lr_4_score = 0.0

# 5. Optimized Weight Blending (All 5 models)
def optimize_weights_fn(probs_to_blend):
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

w_all, opt_all_score = optimize_weights_fn(oof_preds)
print(f"5. Optimized Weights (All 5 models) Balanced Accuracy: {opt_all_score:.6f}  (Weights: {w_all})")

if N_MODELS > 4:
    w_4, opt_4_score = optimize_weights_fn(oof_preds[:, :, :4])
    print(f"6. Optimized Weights (Models 0-3) Balanced Accuracy : {opt_4_score:.6f}  (Weights: {w_4})")
else:
    opt_4_score = 0.0

# Determine the absolute best strategy
strategies = {
    "Simple Average (All 5 models)": (simple_all_score, lambda: le_target.inverse_transform(np.argmax(np.mean(test_preds, axis=2), axis=1))),
    "Simple Average (Models 0-3)": (simple_4_score, lambda: le_target.inverse_transform(np.argmax(np.mean(test_preds[:, :, :4], axis=2), axis=1))),
    "Balanced Meta-LR Stack (All models)": (meta_lr_score, lambda: le_target.inverse_transform(meta_lr.predict(test_meta))),
    "Balanced Meta-LR Stack (Models 0-3)": (meta_lr_4_score, lambda: le_target.inverse_transform(meta_lr_4.predict(test_meta_4))),
    "Optimized Weights (All 5 models)": (opt_all_score, lambda: le_target.inverse_transform(np.argmax(sum(test_preds[:, :, i] * w_all[i] for i in range(N_MODELS)), axis=1))),
    "Optimized Weights (Models 0-3)": (opt_4_score, lambda: le_target.inverse_transform(np.argmax(sum(test_preds[:, :, i] * w_4[i] for i in range(4)), axis=1)))
}

best_strategy_name = max(strategies, key=lambda k: strategies[k][0])
best_score, get_labels_fn = strategies[best_strategy_name]

banner(f"Selected Strategy: {best_strategy_name}")
print(f"OOF Balanced Accuracy: {best_score:.6f}")

final_labels = get_labels_fn()

# ── 6. SUBMISSION ─────────────────────────────────────────────
banner("PHASE 6 — Generating Ultimate Submission")

sub_df = pd.DataFrame({'id': test_ids.values, 'health_condition': final_labels})
ts   = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
name = f'submission_Kaggle_ULTIMATE_OOF_Stack_{ts}.csv'
path = SUBMIT / name
sub_df.to_csv(path, index=False)

print("Class Breakdown:")
u, c = np.unique(final_labels, return_counts=True)
for cls, cnt in zip(u, c):
    print(f"  {cls:<15}: {cnt:>7,}  ({cnt/len(final_labels)*100:.1f}%)")

print(f"\n🏆 SUBMISSION READY!")
print(f"   Strategy : {best_strategy_name}")
print(f"   OOF BalAcc: {best_score:.6f}")
print(f"   File     : {name}")
print(f"   Path     : {path}")
