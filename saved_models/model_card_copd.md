# Model Card: COPD Gold Stage Classifier (CatBoost)

## Model Overview
This model is a CatBoost classifier designed to categorize patients into one of four COPD Gold Stages (Stage 1 to Stage 4) based on demographics, smoking history, and lung function features.

## Intended Use
- **Clinical Triage**: Used to screen and stage patient cases in the Clinical Report Portal.
- **Support Tool**: Designed as a decision support aid, not a standalone diagnostic system.

## Deployed Staging Logic (GOLD Override Layer)
> [!IMPORTANT]
> **Deployed Staging Behavior**
> To address dataset label noise and ensure absolute clinical safety, the staging output in the Patient Triage Portal utilizes a **deterministic GOLD spirometric staging override layer**.
> - **Override Status**: The override layer is **100% active** for all valid portal submissions. Because age, height, gender, FEV1, and FVC parameters are required fields, the clinical override will evaluate and correct every COPD prediction.
> - **Primary Staging**: The final stage returned to the clinician is determined strictly by the override layer.
> - **Internal ML Signal**: The CatBoost model's reported metrics (94.85% accuracy, 0.9146 F1) describe the *internal raw signal* evaluated on the spreadsheet dataset. The raw ML predictions are surfaced as secondary/debug fields (`raw_class` and `raw_confidence`) to assist clinicians as decision support.

## Features & Inputs
The model uses the following 8 features:
1. **Age** (years)
2. **Gender** (1 = Male, 0 = Female)
3. **Smoker** (1 = Yes, 0 = No)
4. **BMI** (kg/m²)
5. **Baseline SpO2** (%)
6. **Baseline Dyspnea (mMRC)** (numeric 1-5 scale)
7. **fev1_pct_predicted** (Baseline FEV1 divided by predicted FEV1, expressed as a percentage)
8. **fev1_fvc_ratio** (Baseline FEV1 divided by predicted FVC)

## Race/Ethnicity Handling & Baseline Selection
> [!NOTE]
> **Race/Ethnicity Customization & Fallback Strategy**
> - **Inference Behavior**: An optional "Race/Ethnicity" field is available in the web portal UI. If specified, the system applies the exact race-specific Hankinson et al. (1999) reference equations for Caucasian, African-American, or Mexican-American populations.
> - **Training and Unknown Fallback**: The raw training dataset (`COPD_Dataset.xlsx`) contains no race/ethnicity data. To prevent train-serve feature skew, the training features are computed using the **Race-Neutral Average** (the mean of the Caucasian, African-American, and Mexican-American predicted values for a given age, gender, and height). If the race is left blank or marked "Other / Unknown" during inference, it defaults to this same Race-Neutral Average.
> - **Pediatric Subjects**: Min age in the cohort is 40. Therefore, only adult reference equations are implemented.

> [!IMPORTANT]
> **Race-Adjustment Clinical Controversy & Fairness Warning**
> - **Clinical Debate**: The practice of using race-specific reference equations is the subject of active debate and reform in pulmonology (e.g., ATS 2023 recommendations).
> - **Sensitivity Disparities**: Using race-specific norms means that a Black patient with the exact same raw spirometry values as a Caucasian patient of the same age, gender, and height may be staged lower (i.e. categorized as having milder disease), because their values are compared against a lower expected baseline. For example, our integration tests showed that an absolute FEV1 of `1.0L` is staged as **Stage 3 (Severe)** under Caucasian norms but **Stage 2 (Moderate)** under African-American norms.
> - **Diagnostic Impact**: The choice between using specific race/ethnicity equations versus a race-neutral average default has direct, unequal effects on clinical diagnostic sensitivity across patient groups, potentially leading to under-diagnosis or under-treatment of minority populations if not interpreted with clinical awareness.

## Root Cause Analysis of Prior Mismatched Coefficients
- **Finding**: The coefficients in the inherited repository code (e.g., Male FEV1 intercept `0.825` and FVC intercept `0.243`) did not correspond to any official Hankinson 1999 tables.
- **Root Cause**: These coefficients actually originated from a 2023 study by Douglas Clark Johnson and Bradford Gardner Johnson (*"Spirometry Reference Equations Including Existing and Novel Parameters"*, *The Open Respiratory Medicine Journal*), where the authors derived custom regression curves from the NHANES III healthy cohort. The specific coefficients copied into our inherited code corresponded to the Black/African-American tables of the Johnson 2023 paper and had been mislabeled in our codebase as the official Hankinson 1999 values.
- **Correction**: We have replaced all equation coefficients with the official, peer-reviewed tables published in:
  *Hankinson JL, Odencrantz JR, Fedan KB. Spirometric reference values from a sample of the general U.S. population. Am J Respir Crit Care Med. 1999;159:179-187 (Tables 4, 5, 6).*

## Corrected Performance (Held-out 20% Test Split)
- **Accuracy**: 94.85%
- **Macro F1-Score**: 0.9146
- **Macro AUROC**: 0.9828

### Confusion Matrix
```
             Predicted Stage 1   Predicted Stage 2   Predicted Stage 3   Predicted Stage 4
True Stage 1        51                   4                   3                   2
True Stage 2         0                 270                   1                   0
True Stage 3         3                   3                  54                   2
True Stage 4         3                   0                   2                  49
```

## Corrected Performance (5-Fold Stratified Cross-Validation)
- **Accuracy**: 93.78% ± 1.19%
- **Macro F1-Score**: 89.93% ± 1.72%
- **Macro AUROC**: 97.68% ± 0.64%
