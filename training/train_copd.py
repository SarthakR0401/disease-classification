import os
import re
import pandas as pd
import numpy as np
import joblib
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "Datasets", "COPD_Dataset.xlsx")
MODELS_DIR = os.path.join(BASE_DIR, "saved_models")

# 1. Load Data (Sheet1 only, excluding corrupted Sheet2 data)
df1 = pd.read_excel(DATA_PATH, sheet_name='Sheet1')

if 'Patient ID' in df1.columns:
    df = df1.drop(columns=['Patient ID'])
else:
    df = df1.copy()

# 2. Preprocess Data
df_proc = df.copy()

# Extract numeric dyspnea
df_proc['Baseline Dyspnea (mMRC)'] = (
    df_proc['Baseline Dyspnea (mMRC)']
    .astype(str).str.extract(r'(\d+)').astype(float)
)

# Encode Gender & Smoker
df_proc['Gender'] = df_proc['Gender'].map({'Male': 1, 'Female': 0})
df_proc['Smoker'] = df_proc['Smoker'].map({'Yes': 1, 'No': 0})

# Encode Target
stage_map = {'Stage 1': 0, 'Stage 2': 1, 'Stage 3': 2, 'Stage 4': 3}
df_proc['target'] = df_proc['COPD Gold Stage'].map(stage_map)
df_proc = df_proc.drop(columns=['COPD Gold Stage'])

# 3. Feature Engineering (NHANES III Reference Equations)
def get_predicted_fev1(age, gender, height_cm, race="Unknown"):
    is_male = (gender == 1)
    coefs = {
        "Male": {
            "Caucasian": [0.5536, -0.01303, -0.000172, 0.00014098],
            "African-American": [0.3411, -0.02309, 0.0, 0.00013194],
            "Mexican-American": [0.6306, -0.02928, 0.0, 0.00015104]
        },
        "Female": {
            "Caucasian": [0.4333, -0.00361, -0.000194, 0.00011496],
            "African-American": [0.3433, -0.01283, -0.000097, 0.00010846],
            "Mexican-American": [0.4529, -0.01178, -0.000113, 0.00012154]
        }
    }
    gender_key = "Male" if is_male else "Female"
    if race in ["Caucasian", "African-American", "Mexican-American"]:
        b0, b1, b2, b3 = coefs[gender_key][race]
        return b0 + b1 * age + b2 * (age ** 2) + b3 * (height_cm ** 2)
    else:
        vals = []
        for r in ["Caucasian", "African-American", "Mexican-American"]:
            b0, b1, b2, b3 = coefs[gender_key][r]
            vals.append(b0 + b1 * age + b2 * (age ** 2) + b3 * (height_cm ** 2))
        return sum(vals) / len(vals)


def get_predicted_fvc(age, gender, height_cm, race="Unknown"):
    is_male = (gender == 1)
    coefs = {
        "Male": {
            "Caucasian": [-0.1933, 0.00064, -0.000269, 0.00018642],
            "African-American": [-0.1517, -0.01821, 0.0, 0.00016643],
            "Mexican-American": [0.2376, -0.00891, -0.000182, 0.00017823]
        },
        "Female": {
            "Caucasian": [-0.3560, 0.01870, -0.000382, 0.00014815],
            "African-American": [-0.3039, 0.00536, -0.000265, 0.00013606],
            "Mexican-American": [0.1210, 0.00307, -0.000237, 0.00014246]
        }
    }
    gender_key = "Male" if is_male else "Female"
    if race in ["Caucasian", "African-American", "Mexican-American"]:
        b0, b1, b2, b3 = coefs[gender_key][race]
        return b0 + b1 * age + b2 * (age ** 2) + b3 * (height_cm ** 2)
    else:
        vals = []
        for r in ["Caucasian", "African-American", "Mexican-American"]:
            b0, b1, b2, b3 = coefs[gender_key][r]
            vals.append(b0 + b1 * age + b2 * (age ** 2) + b3 * (height_cm ** 2))
        return sum(vals) / len(vals)

# Calculate predicted spirometry (using Race-Neutral Average as training baseline)
df_proc['FEV1_pred'] = df_proc.apply(lambda row: get_predicted_fev1(row['Age'], row['Gender'], row['Heigt(cm)']), axis=1)
df_proc['FVC_pred'] = df_proc.apply(lambda row: get_predicted_fvc(row['Age'], row['Gender'], row['Heigt(cm)']), axis=1)

# Engineer features
df_proc['fev1_pct_predicted'] = (df_proc['Baseline FEV1'] / df_proc['FEV1_pred']) * 100
df_proc['fev1_fvc_ratio'] = df_proc['Baseline FEV1'] / df_proc['FVC_pred']

# Features to keep
FEATURES = [
    'Age',
    'Gender',
    'Smoker',
    'BMI',
    'Baseline SpO2',
    'Baseline Dyspnea (mMRC)',
    'fev1_pct_predicted',
    'fev1_fvc_ratio'
]

X = df_proc[FEATURES]
y = df_proc['target']

# Drop rows where target is missing
missing_targets = y.isnull()
if missing_targets.sum() > 0:
    X = X[~missing_targets]
    y = y[~missing_targets]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Impute missing values with training median (leakage-safe)
for col in FEATURES:
    if X_train[col].isnull().sum() > 0:
        median_val = X_train[col].median()
        X_train[col] = X_train[col].fillna(median_val)
        X_test[col] = X_test[col].fillna(median_val)

# 4. Train Model
model = CatBoostClassifier(
    iterations=300,
    random_seed=42,
    verbose=0,
    auto_class_weights='Balanced'
)
model.fit(X_train, y_train)

# 5. Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Stage 1', 'Stage 2', 'Stage 3', 'Stage 4']))

# Feature Importances
importances = model.get_feature_importance()
feat_imp = pd.Series(importances, index=FEATURES).sort_values(ascending=False)
print("\nFeature Importances:")
print(feat_imp)

# Save the new model
joblib.dump(model, os.path.join(MODELS_DIR, "copd_cb_model.pkl"))
print("\nNew COPD model saved as copd_cb_model.pkl")
