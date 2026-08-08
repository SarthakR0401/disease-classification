# Model Card: IPF Screening Risk Score Model

## Model Overview
This model is a binary Logistic Regression classifier designed to evaluate a patient's risk of having moderate-to-severe Idiopathic Pulmonary Fibrosis (IPF) based on basic demographics and spirometry.

## Intended Use
- **Screening Triage**: Designed as a screening triage tool in the Clinical Report Portal to flag high-risk patients who require diagnostic follow-up.
- **Support Tool**: Used to estimate the probability of significant pulmonary impairment, not for final clinical staging.

## Features & Inputs
The model uses the following 5 features:
1. **FVC (mL)** (Forced Vital Capacity)
2. **Percent (%)** (FVC predicted percentage)
3. **Age** (years)
4. **Sex** (1 = Male, 0 = Female)
5. **SmokingStatus** (0 = Currently smokes, 1 = Ex-smoker, 2 = Never smoked)

## Target Reframing
- **Staging to Screening**: Because the dataset (`IPF_Dataset.csv`) only contains 5 clinical features, it does not support complex multi-class diagnostic staging. 
- **Definition**: The model is reframed as a binary classifier:
  - **`0` (Low Risk)**: Corresponds to "Mild" severity.
  - **`1` (High Risk)**: Corresponds to "Moderate" or "Severe" severity.
- **Output**: The probability associated with class `1` represents the patient's **Screening Risk Score** (0% to 100%).
- **Non-Circular Target Origin**: We audited the dataset to ensure the severity labels were not trivially derived via thresholding on the FVC % predicted (`Percent`) feature. The values overlap heavily across classes:
  - Mild range: `[60.39%, 150.00%]`
  - Moderate range: `[41.95%, 143.32%]`
  - Severe range: `[27.77%, 68.74%]`
  This confirms the target label represents a complex clinical assessment rather than a simple mathematical threshold of an input feature.

## Preprocessing & Leakage Fix
- **Data Leakage Fix**: The preprocessing pipeline was audited. The scaling parameters (mean and variance) are computed strictly on the training fold and then applied to transform both splits.
- **Zero Patient Overlap**: Checked the row structure of `IPF_Dataset.csv`. The dataset contains exactly 5,000 rows and **5,000 unique patient IDs** (exactly 1 row per patient). This rules out longitudinal patient overlap across the train/test splits, confirming the validation metrics represent true generalization to unseen subjects.
- **Categorical Mappings**: Features are mapped consistently:
  - Sex: `{'Female': 0, 'Male': 1}`
  - SmokingStatus: `{'Currently smokes': 0, 'Ex-smoker': 1, 'Never smoked': 2}`

## Performance (Held-out 20% Test Split)
- **Accuracy**: 93.10%
- **Precision**: 94.72%
- **Recall**: 92.86%
- **F1-Score**: 93.78%
- **AUROC**: 98.37%

### Confusion Matrix
```
                    Predicted Mild (Low Risk)   Predicted Mod/Sev (High Risk)
True Mild                      411                           29
True Moderate/Severe            40                          520
```

## Performance (5-Fold Stratified Cross-Validation)
- **Accuracy**: 93.64% ± 0.43%
- **Precision**: 95.08% ± 0.98%
- **Recall**: 93.50% ± 0.80%
- **F1-Score**: 94.27% ± 0.37%
- **AUROC**: 98.55% ± 0.27%
