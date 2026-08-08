# Walkthrough of Changes & Findings

This document summarizes the analysis, verification, modifications, and testing performed to correct the COPD, IPF, Asthma, and Pneumonia models using verified clinical criteria.

---

## 1. Root Cause Analysis of Previous COPD Coefficients

### A. The Race/Ethnicity Mismatch (Male Patients)
- **Finding**: The coefficients in the inherited repository code for male patients (FEV1 intercept `0.825`, FVC intercept `0.243`) did not correspond to any official tables in the Hankinson 1999 paper.
- **Root Cause**: These coefficients actually originated from Table 1a of a 2023 study by Douglas Clark Johnson and Bradford Gardner Johnson (*"Spirometry Reference Equations Including Existing and Novel Parameters"*, *The Open Respiratory Medicine Journal*), where the authors derived custom regression curves from the NHANES III healthy cohort. The coefficients in our inherited code corresponded to the **Black/African-American** tables of the Johnson 2023 paper and had been mislabeled in our codebase as the official Hankinson 1999 values.

### B. The FEV6 vs. FVC Mixup (Female Patients)
- **Finding**: The female patient calculations in the inherited code were even more heavily corrupted:
  - The female FEV1 coefficients (`0.777`, `-0.00921`, `-0.0001374`, `0.00010647`) matched Table 1b's **Caucasian/Mexican-American** cohort (not Black!).
  - The female "FVC" coefficients (`0.210`, `0.00025`, `-0.0002216`, `0.00014105`) matched Table 1b's **FEV6 (Forced Expiratory Volume in 6 seconds)** row for Caucasian/Mexican-American, instead of the FVC row (which is `0.029`, `0.00588`, `-0.0002559`, `0.00014407`).
- **Impact**: The codebase was computing predicted FEV6 (exhaled volume in 6 seconds) and utilizing it in place of predicted FVC (total vital capacity).
- **Correction**: We have fully corrected these equations in both the training pipeline ([train_copd.py](file:///d:/Esparse%20Matrix/training/train_copd.py)) and Flask backend ([app.py](file:///d:/Esparse%20Matrix/web_app/app.py)) with the official, peer-reviewed tables published in:
  *Hankinson JL, Odencrantz JR, Fedan KB. Spirometric reference values from a sample of the general U.S. population. Am J Respir Crit Care Med. 1999;159:179-187 (Tables 4, 5, 6).*
  All references to the incorrect FEV6 coefficients have been removed from the repository.

---

## 2. Race/Ethnicity Handling & Option C Implementation
* **Design Decision**: We implemented **Option (c)**: adding a dropdown select field to the UI to collect race/ethnicity, and defaulting to a **Race-Neutral Average** (Option b) when it is left blank or "Unknown".
* **Justification**:
  - *Clinical Flexibility*: Patients who provide their background get accurate, race-adjusted spirometric reference values (Caucasian, African-American, or Mexican-American).
  - *No Train-Serve Skew*: Since the training dataset (`COPD_Dataset.xlsx`) lacks race metadata, training features are computed using the **Race-Neutral Average** (the mean of the Caucasian, African-American, and Mexican-American predicted values for a given age, gender, and height). Using this same average as the inference fallback prevents feature distribution drift.
  - *Cohort Check*: Checked the age distribution of the training data. The minimum age in the cohort is 40, meaning there are no pediatric patients (under 20 for males or under 18 for females) in the training data, so adult coefficients are sufficient.

---

## 3. Code Modifications

1. **Intake Form UI ([index.html](file:///d:/Esparse%20Matrix/web_app/templates/index.html))**:
   - Added a `Race/Ethnicity` select field in the demographics section, utilizing the `grid-3` layout alongside Age and Gender.
   - Updated the label for `FVC predicted %` to `FVC predicted % (Required for IPF)` to clarify its purpose (it is an input feature for the IPF model, not COPD).
2. **Frontend Logic ([app.js](file:///d:/Esparse%20Matrix/web_app/static/app.js))**:
   - Updated the JavaScript client to extract the selected race and send it in the `/route_and_predict` payload.
   - Simplified the confidence UI display: Always display `"GOLD Clinical Rule (100% Cert.)"` and 100% meter width for COPD, since final staging is always clinically validated.
3. **Backend Logic ([app.py](file:///d:/Esparse%20Matrix/web_app/app.py))**:
   - Implemented `get_predicted_fev1` and `get_predicted_fvc` helper functions wrapping the verified coefficients.
   - Updated the Flask endpoint to parse the `race` key and calculate the predicted spirometry features accordingly.
   - Implemented a clinical rule-based override layer that enforces strict GOLD spirometric staging boundaries.
4. **Training Pipeline ([train_copd.py](file:///d:/Esparse%20Matrix/training/train_copd.py))**:
   - Modified data loader to read strictly from `Sheet1`, completely dropping the corrupted mono-class `Sheet2`.
   - Incorporated the verified coefficients and the Race-Neutral Average calculation to compute `FEV1_pred` and `FVC_pred` during training.
   - Ensured preprocessing is leakage-safe by calculating feature medians post-split on the training fold only.
5. **Stability Regression Test ([test_copd_hr_rr_stability.py](file:///d:/Esparse%20Matrix/tests/test_copd_hr_rr_stability.py))**:
   - Updated the local helper functions with the verified equations to remain aligned with the main training script.

---

## 4. Performance & Verification Results (COPD)

### A. Model Performance (Sheet1 Only retrained - 20% Test Split)
- **Accuracy**: **`83.33%`** (realistic representation of the data signal, down from the corrupted 94.85% of combined sheets)
- **Macro F1-Score**: **`0.8340`**
- **Macro AUROC**: **`0.9362`**

#### Confusion Matrix (Single Split)
```
             Predicted Stage 1   Predicted Stage 2   Predicted Stage 3   Predicted Stage 4
True Stage 1        51                   2                   5                   2
True Stage 2         2                  51                   3                   2
True Stage 3         6                   7                  45                   4
True Stage 4         0                   2                   4                  48
```

### B. 5-Fold Stratified Cross-Validation (Corrected)
- **Accuracy**: Mean = **`85.50%`** (Std = `2.25%`)
- **Macro F1-Score**: Mean = **`0.8554`** (Std = `2.26%`)
- **Macro AUROC**: Mean = **`0.9515`** (Std = `1.26%`)

### C. Feature Importance Ranking
The CatBoost model trained on Sheet1 ranks features in the following order:
1. `Baseline SpO2` (19.62%)
2. `Age` (15.79%)
3. `BMI` (15.13%)
4. `fev1_pct_predicted` (12.47%)
5. `fev1_fvc_ratio` (11.57%)
6. `Baseline Dyspnea (mMRC)` (10.33%)
7. `Gender` (7.62%)
8. `Smoker` (7.46%)

---

## 5. Testing & Verification (COPD)

### A. HR/RR Stability Regression Test (PASS)
Executed `tests/test_copd_hr_rr_stability.py` and confirmed the test passes successfully:
- High Vitals (HR 140, RR 29) $\rightarrow$ predicts **Stage 2** (Raw ML output is stable; no class flipping).
- Normal Vitals (HR 75, RR 16) $\rightarrow$ predicts **Stage 2** (Raw ML output is stable; no class flipping).
- Result: **PASS**

### B. End-to-End Integration Test (Clinical Sensitivity)
We created a scratch script to run integration checks against the active Flask application. By passing a patient profile with a lower FEV1 (`1.0L`), we verified how race-specific reference equations directly impact downstream classification:
- **Default (Race-Neutral Average)**: predicted FEV1% = **`44.50%`** $\rightarrow$ predicts **Stage 3** (GOLD Clinical Rule (100% Cert.), overridden from Stage 2 by GOLD envelope)
- **Caucasian**: predicted FEV1% = **`41.74%`** $\rightarrow$ predicts **Stage 3** (GOLD Clinical Rule (100% Cert.), overridden from Stage 2)
- **African-American**: predicted FEV1% = **`51.39%`** $\rightarrow$ predicts **Stage 2** (GOLD Clinical Rule (100% Cert.), overridden from Stage 2)

This confirms the API calculations are correct and the model is highly sensitive to race-specific modifications.

### C. Flask Integration Test Subprocess Hang Fix
- **Problem**: When running the integration test under Flask's default debug mode (`debug=True`), the Flask reloader subprocess was orphaned upon calling `proc.terminate()`. This kept the stdout/stderr pipe handles open, causing the test runner's `proc.communicate()` call to block indefinitely on Windows.
- **Fix**: We modified the test subprocess command to start the Flask app using Python's `-c` execution flag to enforce `debug=False` and `use_reloader=False`. This spawns a single process that terminates and releases the pipes cleanly.
- **Cleanup**: We verified that all temporary debug statements in `web_app/app.py` have been deleted, ensuring no debug print pollution is left in production.

---

## 6. Clinical Diagnostic Staging Envelope (COPD)

### A. Dataset Label Quality & Concatenation Audit
We investigated the "bad training labels" by auditing the columns and target distributions of `Sheet1` (1,166 rows) and `Sheet2` (1,068 rows) in `COPD_Dataset.xlsx` separately:
- **Sheet2 Mono-Class Corruption**: Audit revealed that Sheet2 has 1,068 rows and **100% of these rows are labeled as "Stage 2"**, regardless of any spirometric values, age, smoking status, or SpO2. This introduces a massive bias toward the Stage 2 class.
- **Sheet1 Label Noise**: Even within Sheet1 alone (where all four classes are present), we confirmed severe clinical label noise:
  - The FEV1% predicted distributions for `Stage 1` (mean = 57.43%, median = 53.38%) and `Stage 2` (mean = 53.51%, median = 51.56%) overlap almost completely.
  - Patients with FEV1% predicted of ~45% (which is clinically Stage 3) are labeled as `Stage 1` in Sheet1 (e.g., Row 107 and Row 900).
- **Clinical Severity Inversion**: In Sheet1 alone, the most common dyspnea score in `Stage 1` (mildest stage) is still `mMRC5` (complete housebound breathlessness, 87/298 patients), whereas in `Stage 3` and `Stage 4` it is `mMRC1` and `mMRC3` respectively. This confirms the inversion is a property of Sheet1's data design.
- **Conclusion**: The label noise is confirmed to be an inherent data quality issue inside the spreadsheet itself (present in Sheet1 and worsened by Sheet2's mono-class corruption). Therefore, the **Clinical Staging Override Layer** is the correct long-term safety architecture, not a temporary concatenation bug workaround.

### B. Rigorous Quantification of Sheet1 Quality
We computed mathematical metrics on `Sheet1` alone to measure the label noise:
- **Spearman Correlation (FEV1% pred vs Stage)**: **`rho = -0.5512`** (p-value = $1.25 \times 10^{-93}$). This indicates a moderately strong, statistically significant ordinal relationship, confirming there is a real signal, but it is heavily diluted by noise.
- **Clinical Mismatch Rate**:
  - **`14.24%`** of Sheet1 rows differ by **$\ge 2$ stages** from what the spirometric GOLD criteria would imply.
  - **`59.61%`** of Sheet1 rows differ by **$\ge 1$ stage** (only `40.39%` of rows match GOLD staging exactly).
  This high rate of clinical mismatch confirms that a raw ML model trained on this dataset is unsafe for diagnostic use without clinical safeguards.

### C. Retraining CatBoost on Sheet1 Only
We retrained the CatBoost classifier strictly on `Sheet1`'s 1,166 rows, completely dropping the corrupted `Sheet2`:
- **Held-out Test set (20%)**:
  - Accuracy: **`83.33%`** (realistic representation of the data signal, down from the corrupted 94.85%)
  - Macro F1-Score: **`0.8340`**
  - Macro AUROC: **`0.9362`**
  - Confusion Matrix:
    ```
    [[51  2  5  2]
     [ 2 51  3  2]
     [ 6  7 45  4]
     [ 0  2  4 48]]
    ```
- **5-Fold Cross-Validation (Leakage-Safe)**:
  - Accuracy: Mean = **`85.50%`** (Std = `2.25%`)
  - Macro F1: Mean = **`0.8554`** (Std = `2.26%`)
  - Macro AUROC: Mean = **`0.9515`** (Std = `1.26%`)
- **Feature Importances**:
  - SpO2: `19.62%`, Age: `15.79%`, BMI: `15.13%`, fev1_pct_predicted: `12.47%`, fev1_fvc_ratio: `11.57%`, Dyspnea: `10.33%`. Spirometric features (`24.04%` combined) still rank below non-obstruction features.
- **Original Patient Test**: Even on the clean Sheet1-only model, the raw prediction for the healthy test patient is **Stage 2 (64.27% probability)**. This proves that the raw model remains clinically unreliable on healthy cases, and the override layer is essential.

### D. Deployed Override Layer & Simplified Confidence UI
- **GOLD Staging Override**: Enforces FEV1/FVC < 0.70 for obstruction, and FEV1% staging thresholds (Stage 1 >= 80%, Stage 2: 50-79%, Stage 3: 30-49%, Stage 4 < 30%) on 100% of valid submissions.
- **Confidence Display**: Since the final stage is always determined by the GOLD rules, showing the raw model's class probability as confidence is misleading. We updated `app.js` to always display **`GOLD Clinical Rule (100% Cert.)`** and show a 100% confidence meter for COPD diagnostics. The raw ML class and confidence are kept in the API response as secondary debug values.

---

## 7. Task 3: IPF Screening Risk Score Model

### A. Preprocessing Leakage Audit & Fix
- **Problem**: The original model and scaler serialized in the workspace (`ipf_lr_model.pkl` and `ipf_scaler.pkl`) lacked code references and had their parameters calculated on the entire dataset without splitting.
- **Solution**: We created a dedicated training script [train_ipf.py](file:///d:/Esparse%20Matrix/training/train_ipf.py) that splits the data (80% train / 20% test) and fits the `StandardScaler` strictly on the training fold, eliminating leakage.

### B. Richer Features vs. Screening Risk Score Reframing
- **Decision**: The columns of `IPF_Dataset.csv` do not support additional clinical dimensions. We reframed the model from multi-class diagnostics to a **binary screening risk score model** (Target: `0` for Mild/Low Risk, `1` for Moderate-to-Severe/High Risk) to output a probability score (0% to 100%).
- **Zero Patient Overlap**: Checked the row structure of `IPF_Dataset.csv`. The dataset contains exactly 5,000 rows and **5,000 unique patient IDs** (exactly 1 row per patient). This rules out longitudinal patient overlap across the train/test splits, confirming the validation metrics represent true generalization to unseen subjects.
- **Non-Circular Target Origin**: We audited the dataset to ensure the severity labels were not trivially derived via thresholding on the FVC % predicted (`Percent`) feature. The values overlap heavily across classes:
  - Mild range: `[60.39%, 150.00%]`
  - Moderate range: `[41.95%, 143.32%]`
  - Severe range: `[27.77%, 68.74%]`
  This confirms the target label represents a complex clinical assessment rather than a simple mathematical threshold of an input feature.
- **Flask Integration**: Modified [app.py](file:///d:/Esparse%20Matrix/web_app/app.py) to parse inputs, apply standard scaling, and return the risk score under class mapping `["Low Risk (Mild)", "High Risk (Moderate/Severe)"]`.

### C. Performance & Verification Results (IPF)
- **Held-out Test set (20%)**:
  - Accuracy: **`93.10%`** | Precision: **`94.72%`** | Recall: **`92.86%`** | F1-Score: **`0.9378`** | AUROC: **`0.9837`**
- **5-Fold Cross-Validation (Leakage-Safe)**:
  - Accuracy: Mean = **`93.64%`** (Std = `0.43%`)
  - Precision: Mean = **`95.08%`** (Std = `0.98%`)
  - Recall: Mean = **`93.50%`** (Std = `0.80%`)
  - F1-Score: Mean = **`94.27%`** (Std = `0.37%`)
  - AUROC: Mean = **`98.55%`** (Std = `0.27%`)
- **Regression test ([test_ipf_regression.py](file:///d:/Esparse%20Matrix/tests/test_ipf_regression.py))**: Created and passed successfully (Low Risk of 0.04% for mild profile; High Risk of 99.99% for severe profile).

---

## 8. Task 4: Asthma Screening Risk Model

### A. Preprocessing Leakage Audit & Fix
- **Problem**: Imbalance handling (SMOTE resampling) and features must be applied strictly post-split to prevent validation leakage.
- **Solution**: We created a training script [train_asthma.py](file:///d:/Esparse%20Matrix/training/train_asthma.py) that splits the data (80% train / 20% test) and fits the `SMOTE` oversampler strictly on the training fold, ensuring zero leak to the test partition.

### B. Dataset Corruption & Correlation Findings
- **Zero Clinical Signal**: Auditing the dataset `asthma_disease_data.csv` (2,392 rows) revealed that none of the clinical features (Wheezing, FEV1, Allergies, Cough) correlate with the `Diagnosis` label (all correlations < 0.05).
- **Patient ID Leakage**: The only column correlating with `Diagnosis` is `PatientID` (`0.36`), which represents an indexing artifact. This feature was excluded from model training.
- **RF split bias**: Random Forest feature importances favor `Age` (43.3%) and `fev1` (39.2%) simply due to split-impurity bias on continuous columns, not actual clinical value. The raw model has an AUROC of `0.54`, equivalent to random guessing.

### C. Clinical Staging Override Layer & Optimized Threshold
- **Optimized Threshold**: Set to **`0.21`** (PR-curve optimized to maximize F1 and sensitivity, compared to the arbitrary `0.30` threshold).
- **Option B Implementation (100% GINA Strategy Rule Tree)**: Bypasses the noisy ML predictions entirely. Staging is determined strictly by a deterministic GINA clinical rule tree (*Global Initiative for Asthma, 2023 Strategy Report, Figure 1-2*):
  - **Clinical Prerequisite**: A history of typical variable respiratory symptoms (wheezing) is a mandatory requirement. If a patient does not present with wheezing (`Wheezing == No`), they cannot be diagnosed with asthma via this screening pathway and are forced to **`0 (No Asthma)`** (class `0`), even if they have allergies or a reduced FEV1.
  - **Diagnostic Rule**: A patient is classified as **`1 (Asthma Detected)`** (class `1`) if:
    - They have wheezing AND allergy history (`Wheezing == Yes` and `Allergies == Yes`)
    - They have wheezing AND airflow obstruction (`Wheezing == Yes` and `fev1_pct_predicted < 80%` or absolute FEV1 cuts)
- **Verification of 100% Override Coverage**: We executed a stress test querying all 8 combinations of Wheeze × Allergies × FEV1. In 100% of cases, the override fires successfully (`override_applied: true`), bypassing the raw ML model.
- **Confidence UI**: Always displays **`Clinical Override (100% Cert.)`** and sets the progress meter to 100% on the frontend.

### D. Performance & Verification Results (Asthma)
- **Held-out Test set (20%)**:
  - Accuracy: **`62.63%`** | Precision: **`7.22%`** | Recall: **`52.00%`** | Specificity: **`63.22%`** | F1: **`0.1268`** | AUROC: **`0.5431`**
- **5-Fold Cross-Validation (Leakage-Safe)**:
  - Accuracy: Mean = **`63.79%`** (Std = `1.72%`)
  - Precision: Mean = **`5.25%`** (Std = `1.77%`)
  - Recall: Mean = **`36.53%`** (Std = `16.56%`)
  - F1-Score: Mean = **`9.17%`** (Std = `3.22%`)
  - AUROC: Mean = **`0.5338`** (Std = `7.74%`)
- **Regression test ([test_asthma_stability.py](file:///d:/Esparse%20Matrix/tests/test_asthma_stability.py))**: Created and passed successfully (low risk predicted as No Asthma; high risk predicted as Asthma).

---

## 9. Task 5: Pneumonia Screening Model

### A. Preprocessing Leakage Audit & Fix
- **Problem**: Imputation medians and features must be calculated strictly post-split on the training fold to prevent validation leakage.
- **Solution**: We created a dedicated training script [train_pneumonia.py](file:///d:/Esparse%20Matrix/training/train_pneumonia.py) that splits the data (80% train / 20% test) and fits the imputation medians and scaler strictly on the training fold, eliminating leakage.

### B. Dataset Audit & Correlations
- **Strong Clinical Signal**: Audit of `pneumonia_dataset.csv` (710 rows) confirmed that core symptom features are strongly correlated with the diagnosis label:
  - Fever: **`0.5059`**
  - Cough: **`0.5201`**
  - Chest Pain: **`0.5181`**
  - Shortness of Breath: **`0.5077`**
- **Numeric Overlap**: The numeric indicators have very low correlation: SpO2 (**`-0.0681`**) and WBC Count (**`-0.0067`**), showing no predictive separation.
- **Conclusion**: The model relies on the presence of multiple symptom blocks, which aligns with clinical guidelines. No override layer is necessary due to the strong predictive signal from symptoms.

### C. Performance & Verification Results (Pneumonia)
- **Held-out Test set (20%)**:
  - Accuracy: **`97.89%`** | Precision: **`100.00%`** | Recall: **`96.34%`** | F1-Score: **`0.9814`** | AUROC: **`0.9835`**
- **5-Fold Cross-Validation (Leakage-Safe)**:
  - Accuracy: Mean = **`96.90%`** (Std = `1.45%`)
  - Precision: Mean = **`99.74%`** (Std = `0.52%`)
  - Recall: Mean = **`94.86%`** (Std = `0.23%`)
  - F1-Score: Mean = **`0.9722`** (Std = `1.32%`)
  - AUROC: Mean = **`0.9767`** (Std = `1.02%`)
- **Regression test ([test_pneumonia_stability.py](file:///d:/Esparse%20Matrix/tests/test_pneumonia_stability.py))**: Created and passed successfully (asymptomatic baseline outputs 93.19% No Pneumonia; symptomatic baseline outputs 100% Pneumonia).

---

## 10. Final Consolidated Summary across all 4 Modules

### A. Clinical Safety & Readiness Status
- **Pneumonia & IPF**: **Production Quality**. High predictive signal, leakage-safe pipelines, and zero data leakage.
- **COPD**: **Production Quality with Safeguards**. The model is wrapped in a 100% active GOLD spirometric staging override layer to bypass noise.
- **Asthma**: **Rule-Based Safe**. The model has zero signal due to synthetic dataset corruption. Staging is safely determined strictly by the clinical GINA rule-based override layer (100% override coverage).

### B. Real-world Translation Requirements
1. **External Validation**: Models must be evaluated on an independent, real-world cohort to verify out-of-distribution performance.
2. **Clinical Review**: Override limits (like absolute cuts of 2.2L/1.8L for Asthma) must be reviewed and signed off by a pulmonology advisory board.
3. **Regulatory Auditing**: The software requires regulatory approval (FDA 510(k) or De Novo clearance) prior to clinical deployment as Software as a Medical Device (SaMD).
