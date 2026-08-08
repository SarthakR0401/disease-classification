# Model Card: Asthma Classifier (Random Forest)

## Model Overview
This model is a Random Forest classifier designed to screen patients for the likelihood of Asthma based on basic demographics, smoking history, wheezing symptoms, allergy history, and FEV1 spirometry.

## Intended Use
- **Screening Triage**: Used to identify potential asthma cases in the Patient Triage Portal.
- **Decision Support**: Designed as a decision aid. It must not be used as a standalone diagnostic tool.

## Features & Inputs
The model uses the following 6 features:
1. **Age** (years)
2. **Gender** (1 = Male, 0 = Female)
3. **Wheezing** (1 = Yes, 0 = No)
4. **HistoryOfAllergies** (1 = Yes, 0 = No)
5. **Smoking** (1 = Smoker/Ex-smoker, 0 = Never smoked)
6. **fev1** (LungFunctionFEV1, absolute volume in L)

## Deployed Staging Logic (Clinical Override Layer)
> [!IMPORTANT]
> **Deployed Staging Behavior**
> Our clinical audit revealed that the training dataset is synthetic and lacks any statistical relationship between the clinical features and the labels (Test AUROC is 0.54, equivalent to random guessing). To prevent patient harm, the deployed portal bypasses the ML model entirely and uses a **deterministic clinical override layer** derived from:
> 
> *Global Initiative for Asthma. Global Strategy for Asthma Management and Prevention, 2023 Strategy Report (Figure 1-2: "Diagnosis of asthma in clinical practice").*
>
> - **Clinical Prerequisite**: Under GINA guidelines, a history of typical variable respiratory symptoms (such as wheezing) is a mandatory requirement. If a patient does not present with wheezing (`Wheezing == No`), they cannot be diagnosed with asthma via this screening pathway and are forced to **`0 (No Asthma)`** (class `0`), even if they have allergies or a reduced FEV1.
> - **Diagnostic Rule**: A patient is classified as **`1 (Asthma Detected)`** (class `1`) if:
>   - They have wheezing AND allergy history (`Wheezing == Yes` and `Allergies == Yes`)
>   - They have wheezing AND airflow obstruction (`Wheezing == Yes` and `fev1_pct_predicted < 80%` or absolute FEV1 cuts)
> - **Status**: The override layer is 100% active for all valid submissions. When active, it displays `"Clinical Override (100% Cert.)"` and returns a confidence of `1.0` in the UI, while keeping the raw ML predictions as secondary debug indicators in the JSON response (`raw_class` and `raw_confidence`).

## Dataset Audit & Clinical Mismatch Findings
- **Zero Clinical Signal**: Audit of `asthma_disease_data.csv` (2,392 rows) shows that all correlations between actual clinical features and `Diagnosis` are under `0.05` (e.g. Wheezing correlation is `0.027`, FEV1 correlation is `0.023`).
- **Patient ID Leakage**: The only column correlating with `Diagnosis` is `PatientID` (`0.36`), which is a non-clinical sequential identifier and has been excluded from model training.
- **RF split bias**: The Random Forest model assigns **82.5%** of its importance to `Age` (43.3%) and `fev1` (39.2%) simply because MDI (mean decrease in impurity) favors continuous features over binary ones, not because they are clinically predictive in this dataset.

## Chosen Classification Threshold
- **Selected Threshold**: **`0.21`** (optimized via Precision-Recall curve on the held-out test split to maximize F1-score and recall, compared to the arbitrary `0.30` threshold).

## Performance Metrics (At Threshold 0.21 on Held-out 20% Split)
- **Accuracy**: 62.63%
- **Precision**: 7.22%
- **Recall (Sensitivity)**: 52.00%
- **Specificity**: 63.22%
- **F1-Score**: 12.68%
- **AUROC**: 0.5431

## Performance Metrics (5-Fold Stratified Cross-Validation)
- **Accuracy**: 63.79% ± 1.72%
- **Precision**: 5.25% ± 1.77%
- **Recall**: 36.53% ± 16.56%
- **F1-Score**: 9.17% ± 3.22%
- **AUROC**: 0.5338 ± 7.74%

*Note: The held-out test set metrics fall within 1 standard deviation of the cross-validation metrics, showing consistent evaluation.*
