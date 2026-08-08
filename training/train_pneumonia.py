import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "datasets", "pneumonia_dataset.csv")
MODELS_DIR = os.path.join(BASE_DIR, "saved_models")

# 1. Load Data
df = pd.read_csv(DATA_PATH)

# Encode Target
df['target'] = df['Diagnosis'].map({'Yes': 1, 'No': 0})

# Encode Categorical features (exactly matching app.py mappings)
fever_map = {"High": 3, "Moderate": 2, "Low": 1}
cough_map = {"Bloody": 3, "Wet": 2, "Dry": 1}
cp_map = {"Severe": 3, "Moderate": 2, "Mild": 1}
rs_map = {"Severe": 3, "Moderate": 2, "Mild": 1}

df['Fever_encoded'] = df['Fever'].map(fever_map).fillna(1)
df['Cough_encoded'] = df['Cough'].map(cough_map).fillna(1)
df['ChestPain_encoded'] = df['Chest_pain'].map(cp_map).fillna(1)
df['RespiratorySymptom_encoded'] = df['Shortness_of_breath'].map(rs_map).fillna(1)

# Clean numeric features
df['SpO2_clean'] = pd.to_numeric(df['Oxygen_saturation'].replace('-', np.nan), errors='coerce')
df['WBCCount_clean'] = pd.to_numeric(df['WBC_count'].replace('-', np.nan), errors='coerce')

FEATURES = ['Fever_encoded', 'Cough_encoded', 'ChestPain_encoded', 'WBCCount_clean', 'SpO2_clean', 'RespiratorySymptom_encoded']
X = df[FEATURES].copy()
y = df['target'].copy()

# 2. Split First
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Leakage-safe Imputation (Fit on Train fold, transform on Test fold)
train_wbc_median = X_train['WBCCount_clean'].median()
train_spo2_median = X_train['SpO2_clean'].median()

X_train['WBCCount_clean'] = X_train['WBCCount_clean'].fillna(train_wbc_median)
X_train['SpO2_clean'] = X_train['SpO2_clean'].fillna(train_spo2_median)

X_test['WBCCount_clean'] = X_test['WBCCount_clean'].fillna(train_wbc_median)
X_test['SpO2_clean'] = X_test['SpO2_clean'].fillna(train_spo2_median)

# 4. Leakage-safe Scaling (Fit on Train fold, transform on Test fold)
scaler = StandardScaler()
scaler.fit(X_train)

# 5. Train Decision Tree
# Max depth 8 matches original bundle, but we train it on unscaled features since decision trees are scale-invariant
model = DecisionTreeClassifier(max_depth=8, random_state=42)
model.fit(X_train.values, y_train.values)

# 6. Evaluate on Held-out test split
test_preds = model.predict(X_test.values)
test_probs = model.predict_proba(X_test.values)[:, 1]

print("=== Held-out Test Set Metrics (Pneumonia) ===")
print(f"Accuracy: {accuracy_score(y_test, test_preds):.4f}")
print(f"Precision: {precision_score(y_test, test_preds):.4f}")
print(f"Recall: {recall_score(y_test, test_preds):.4f}")
print(f"F1-Score: {f1_score(y_test, test_preds):.4f}")
print(f"AUROC: {roc_auc_score(y_test, test_probs):.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, test_preds))

# 7. 5-Fold Stratified Cross-Validation (Leakage-Safe)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_accs, cv_precs, cv_recs, cv_f1s, cv_aurocs = [], [], [], [], []

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
    X_tr, y_tr = X.iloc[train_idx].copy(), y.iloc[train_idx].copy()
    X_te, y_te = X.iloc[test_idx].copy(), y.iloc[test_idx].copy()
    
    # Impute inside fold
    tr_wbc = X_tr['WBCCount_clean'].median()
    tr_spo2 = X_tr['SpO2_clean'].median()
    X_tr['WBCCount_clean'] = X_tr['WBCCount_clean'].fillna(tr_wbc)
    X_tr['SpO2_clean'] = X_tr['SpO2_clean'].fillna(tr_spo2)
    X_te['WBCCount_clean'] = X_te['WBCCount_clean'].fillna(tr_wbc)
    X_te['SpO2_clean'] = X_te['SpO2_clean'].fillna(tr_spo2)
    
    fold_model = DecisionTreeClassifier(max_depth=8, random_state=42)
    fold_model.fit(X_tr.values, y_tr.values)
    
    fold_preds = fold_model.predict(X_te.values)
    fold_probs = fold_model.predict_proba(X_te.values)[:, 1]
    
    cv_accs.append(accuracy_score(y_te, fold_preds))
    cv_precs.append(precision_score(y_te, fold_preds, zero_division=0))
    cv_recs.append(recall_score(y_te, fold_preds, zero_division=0))
    cv_f1s.append(f1_score(y_te, fold_preds, zero_division=0))
    cv_aurocs.append(roc_auc_score(y_te, fold_probs))

print("\n=== 5-Fold Stratified Cross-Validation ===")
print(f"Accuracy: mean = {np.mean(cv_accs):.4f}, std = {np.std(cv_accs):.4f}")
print(f"Precision: mean = {np.mean(cv_precs):.4f}, std = {np.std(cv_precs):.4f}")
print(f"Recall: mean = {np.mean(cv_recs):.4f}, std = {np.std(cv_recs):.4f}")
print(f"F1-Score: mean = {np.mean(cv_f1s):.4f}, std = {np.std(cv_f1s):.4f}")
print(f"AUROC: mean = {np.mean(cv_aurocs):.4f}, std = {np.std(cv_aurocs):.4f}")

# 8. Save updated bundle
bundle = {
    "best_model_name": "DecisionTree",
    "best_model": model,
    "features": ['Fever', 'Cough', 'ChestPain', 'WBCCount', 'SpO2', 'RespiratorySymptom'],
    "scaler": scaler,
    "imputation_values": {
        "Fever": 1.0,
        "Cough": 1.0,
        "ChestPain": 1.0,
        "RespiratorySymptom": 1.0,
        "WBCCount": float(train_wbc_median),
        "SpO2": float(train_spo2_median)
    },
    "categorical_features": ['Fever', 'Cough', 'ChestPain', 'RespiratorySymptom'],
    "numeric_features": ['WBCCount', 'SpO2'],
    "class_labels": {0: "No Pneumonia", 1: "Pneumonia"}
}
joblib.dump(bundle, os.path.join(MODELS_DIR, "pneumonia_bundle.pkl"))
print("\nSaved updated Pneumonia bundle to saved_models/pneumonia_bundle.pkl")
