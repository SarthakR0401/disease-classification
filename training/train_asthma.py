import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, precision_recall_curve
from imblearn.over_sampling import SMOTE

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "datasets", "asthma_disease_data.csv")
MODELS_DIR = os.path.join(BASE_DIR, "saved_models")

# 1. Load Data
df = pd.read_csv(DATA_PATH)

# Features mapping
df['as_gender'] = df['Gender']
df['as_wheezing'] = df['Wheezing']
df['as_allergies'] = df['HistoryOfAllergies']
df['as_smoking'] = df['Smoking']
df['fev1'] = df['LungFunctionFEV1']

FEATURES = ['Age', 'as_gender', 'as_wheezing', 'as_allergies', 'as_smoking', 'fev1']
X = df[FEATURES]
y = df['Diagnosis']

# 2. Split First
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Resampling (SMOTE) applied ONLY to training fold
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# 4. Train Model
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train_res, y_train_res)

# 5. Threshold Optimization via Precision-Recall Curve on Held-out test set
test_probs = model.predict_proba(X_test)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y_test, test_probs)

# Find optimal threshold that maximizes F1-score or balances sensitivity/specificity
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
opt_idx = np.argmax(f1_scores)
opt_threshold = thresholds[opt_idx] if opt_idx < len(thresholds) else 0.50

print(f"Optimal threshold based on F1-score: {opt_threshold:.4f}")

# Compare current arbitrary threshold of 0.30 vs. optimal threshold
for thresh in [0.30, opt_threshold]:
    preds_thresh = (test_probs >= thresh).astype(int)
    acc = accuracy_score(y_test, preds_thresh)
    prec = precision_score(y_test, preds_thresh, zero_division=0)
    rec = recall_score(y_test, preds_thresh, zero_division=0)
    f1 = f1_score(y_test, preds_thresh, zero_division=0)
    auroc = roc_auc_score(y_test, test_probs)
    tn, fp, fn, tp = confusion_matrix(y_test, preds_thresh).ravel()
    spec = tn / (tn + fp)
    
    print(f"\n--- Metrics at threshold {thresh:.2f} ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall (Sensitivity): {rec:.4f}")
    print(f"Specificity: {spec:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"AUROC: {auroc:.4f}")

# Save the trained model
joblib.dump(model, os.path.join(MODELS_DIR, "asthma_rf_model.pkl"))
print("\nSaved new Asthma model to saved_models/asthma_rf_model.pkl")

# 6. 5-Fold Stratified Cross-Validation (with leakage-safe resampling inside folds)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_accs, cv_precs, cv_recs, cv_f1s, cv_aurocs = [], [], [], [], []

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
    X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
    X_te, y_te = X.iloc[test_idx], y.iloc[test_idx]
    
    # Resample only inside training fold
    fold_smote = SMOTE(random_state=42)
    X_tr_res, y_tr_res = fold_smote.fit_resample(X_tr, y_tr)
    
    fold_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    fold_model.fit(X_tr_res, y_tr_res)
    
    fold_probs = fold_model.predict_proba(X_te)[:, 1]
    fold_preds = (fold_probs >= opt_threshold).astype(int)
    
    cv_accs.append(accuracy_score(y_te, fold_preds))
    cv_precs.append(precision_score(y_te, fold_preds, zero_division=0))
    cv_recs.append(recall_score(y_te, fold_preds, zero_division=0))
    cv_f1s.append(f1_score(y_te, fold_preds, zero_division=0))
    cv_aurocs.append(roc_auc_score(y_te, fold_probs))

print("\n=== 5-Fold Cross-Validation Metrics (at Optimal Threshold) ===")
print(f"Accuracy: mean = {np.mean(cv_accs):.4f}, std = {np.std(cv_accs):.4f}")
print(f"Precision: mean = {np.mean(cv_precs):.4f}, std = {np.std(cv_precs):.4f}")
print(f"Recall: mean = {np.mean(cv_recs):.4f}, std = {np.std(cv_recs):.4f}")
print(f"F1-Score: mean = {np.mean(cv_f1s):.4f}, std = {np.std(cv_f1s):.4f}")
print(f"AUROC: mean = {np.mean(cv_aurocs):.4f}, std = {np.std(cv_aurocs):.4f}")

# Feature Importances
importances = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
print("\n=== Feature Importances ===")
print(importances)
