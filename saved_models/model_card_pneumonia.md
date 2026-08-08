# Model Card: Pneumonia Classifier (Decision Tree)

## Model Overview
This model is a Decision Tree classifier designed to screen patient cases for Pneumonia based on basic symptoms (fever, cough, chest pain, shortness of breath), WBC count, and SpO2 oxygen saturation.

## Intended Use
- **Screening Triage**: Used to identify potential pneumonia cases in the Patient Triage Portal.
- **Decision Support**: Designed as a decision aid. It must not be used as a standalone diagnostic tool.

## Features & Inputs
The model uses the following 6 features:
1. **Fever** (encoded 1-3: Low/No = 1, Moderate = 2, High = 3)
2. **Cough** (encoded 1-3: Dry/No = 1, Wet = 2, Bloody = 3)
3. **ChestPain** (encoded 1-3: Mild/No = 1, Moderate = 2, Severe = 3)
4. **WBCCount** (numeric, white blood cell count in cells/mcL)
5. **SpO2** (numeric, baseline oxygen saturation in %)
6. **RespiratorySymptom** (Shortness of breath, encoded 1-3: Mild/No = 1, Moderate = 2, Severe = 3)

## Dataset Audit & Clinical Integrity
- **Strong Symptom Correlations**: Auditing `pneumonia_dataset.csv` (710 rows) revealed that symptom features are strongly correlated with the diagnosis label:
  - `Fever`: **`0.5059`**
  - `Cough`: **`0.5201`**
  - `Chest Pain`: **`0.5181`**
  - `Shortness of Breath`: **`0.5077`**
- **Numeric Overlap**: The numeric indicators have very low correlation with diagnosis:
  - `Oxygen_saturation` (SpO2): **`-0.0681`** (Mean SpO2 is 92.4% for pneumonia vs 93.0% for healthy, representing highly overlapping distributions)
  - `WBC_count`: **`-0.0067`** (Mean WBC is 9234 vs 9553, showing no predictive separation)
- **Conclusion**: The model relies heavily on the presence of multiple symptom blocks (fever, cough, chest pain, SOB) to establish high-confidence predictions, which aligns with clinical screening guidelines. No clinical override layer is necessary due to the strong predictive signal from symptom features.

## Preprocessing & Imputation (Leakage-Safe)
- **Split-First Partitioning**: Data is split into train and test sets before any preprocessing.
- **Median Imputation**: Imputation values are calculated strictly on the training partition:
  - `WBCCount` fallback median: **`9800`**
  - `SpO2` fallback median: **`93.0`**
- **Scaler**: A `StandardScaler` is fitted on training data and stored in the bundle, though the Decision Tree splits are scale-invariant.

## Performance Metrics (Held-out 20% Test Split)
- **Accuracy**: 97.89%
- **Precision**: 100.00%
- **Recall (Sensitivity)**: 96.34%
- **F1-Score**: 98.14%
- **AUROC**: 0.9835

### Confusion Matrix
```
                 Predicted No Pneumonia   Predicted Pneumonia
True No Pneumonia         60                       0
True Pneumonia             3                      79
```

## Performance Metrics (5-Fold Stratified Cross-Validation)
- **Accuracy**: 96.90% ± 1.45%
- **Precision**: 99.74% ± 0.52%
- **Recall**: 94.86% ± 2.36%
- **F1-Score**: 97.22% ± 1.32%
- **AUROC**: 0.9767 ± 1.02%

*Note: The held-out test split performance falls within 1 standard deviation of the cross-validation metrics.*
