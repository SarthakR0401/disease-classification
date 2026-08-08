import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "datasets", "IPF_Dataset.csv")
MODELS_DIR = os.path.join(BASE_DIR, "saved_models")

# 1. Load Data
df = pd.read_csv(DATA_PATH)

# Encode Sex & SmokingStatus
df['Sex_encoded'] = df['Sex'].map({'Male': 1, 'Female': 0})
df['Smoking_encoded'] = df['SmokingStatus'].map({
    'Currently smokes': 0,
    'Ex-smoker': 1,
    'Never smoked': 2
})

# Reframe target: 1 = Moderate/Severe (High Screening Risk), 0 = Mild (Low Screening Risk)
df['target'] = df['IPF_Severity'].map({'Mild': 0, 'Moderate': 1, 'Severe': 1})

FEATURES = ['FVC', 'Percent', 'Age', 'Sex_encoded', 'Smoking_encoded']
X = df[FEATURES]
y = df['target']

# 2. Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Leakage-safe scaling (fit ONLY on training fold, transform both)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler and encoders for inference
joblib.dump(scaler, os.path.join(MODELS_DIR, "ipf_scaler.pkl"))
encoders = {
    'Sex': {'Male': 1, 'Female': 0},
    'SmokingStatus': {'Currently smokes': 0, 'Ex-smoker': 1, 'Never smoked': 2},
    'IPF_Severity': {'Mild': 0, 'Moderate': 1, 'Severe': 1}
}
joblib.dump(encoders, os.path.join(MODELS_DIR, "ipf_label_encoders.pkl"))

# 4. Train Model
model = LogisticRegression(random_state=42, class_weight='balanced')
model.fit(X_train_scaled, y_train)

# 5. Evaluate on Held-out test set
y_pred = model.predict(X_test_scaled)
probs = model.predict_proba(X_test_scaled)[:, 1]

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auroc = roc_auc_score(y_test, probs)

print("=== Held-out Test Set Metrics ===")
print(f"Accuracy: {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall: {rec:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"AUROC: {auroc:.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save the trained model
joblib.dump(model, os.path.join(MODELS_DIR, "ipf_lr_model.pkl"))
print(f"\nSaved new IPF screening model to saved_models/ipf_lr_model.pkl")

# 6. 5-Fold Stratified Cross-Validation (with leakage-safe folding)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_accs, cv_precs, cv_recs, cv_f1s, cv_aurocs = [], [], [], [], []

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
    X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
    X_te, y_te = X.iloc[test_idx], y.iloc[test_idx]
    
    # Scale inside each fold independently
    fold_scaler = StandardScaler()
    X_tr_scaled = fold_scaler.fit_transform(X_tr)
    X_te_scaled = fold_scaler.transform(X_te)
    
    fold_model = LogisticRegression(random_state=42, class_weight='balanced')
    fold_model.fit(X_tr_scaled, y_tr)
    
    preds_fold = fold_model.predict(X_te_scaled)
    probs_fold = fold_model.predict_proba(X_te_scaled)[:, 1]
    
    cv_accs.append(accuracy_score(y_te, preds_fold))
    cv_precs.append(precision_score(y_te, preds_fold))
    cv_recs.append(recall_score(y_te, preds_fold))
    cv_f1s.append(f1_score(y_te, preds_fold))
    cv_aurocs.append(roc_auc_score(y_te, probs_fold))

print("\n=== 5-Fold Cross-Validation Metrics ===")
print(f"Accuracy: mean = {np.mean(cv_accs):.4f}, std = {np.std(cv_accs):.4f}")
print(f"Precision: mean = {np.mean(cv_precs):.4f}, std = {np.std(cv_precs):.4f}")
print(f"Recall: mean = {np.mean(cv_recs):.4f}, std = {np.std(cv_recs):.4f}")
print(f"F1-Score: mean = {np.mean(cv_f1s):.4f}, std = {np.std(cv_f1s):.4f}")
print(f"AUROC: mean = {np.mean(cv_aurocs):.4f}, std = {np.std(cv_aurocs):.4f}")
