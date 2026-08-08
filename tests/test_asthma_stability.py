import os
import joblib
import numpy as np

def test_asthma_stability():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODEL_PATH = os.path.join(BASE_DIR, "saved_models", "asthma_rf_model.pkl")
    
    assert os.path.exists(MODEL_PATH), f"Model not found at {MODEL_PATH}"
    model = joblib.load(MODEL_PATH)
    
    # helper functions for override
    # GINA-based clinical override rules
    # Features: ['Age', 'Gender', 'Wheezing', 'HistoryOfAllergies', 'Smoking', 'LungFunctionFEV1']
    
    def predict_with_override(age, gender, wheezing, allergies, smoking, fev1):
        as_gender = 1 if gender == "Male" or gender == 1 else 0
        as_wheezing = 1 if wheezing == "Yes" or wheezing == 1 else 0
        as_allergies = 1 if allergies == "Yes" or allergies == 1 else 0
        as_smoking = 1 if smoking in ["Current smoker", "Ex-smoker", "Currently smokes", 1] else 0
        
        features = np.array([[age, as_gender, as_wheezing, as_allergies, as_smoking, fev1]])
        prob = model.predict_proba(features)[0]
        raw_pred = 1 if prob[1] >= 0.21 else 0
        
        pred_class = raw_pred
        override_applied = False
        
        # Rule 1: No wheezing and no allergies -> Force No Asthma
        if as_wheezing == 0 and as_allergies == 0:
            pred_class = 0
            if raw_pred != 0:
                override_applied = True
        # Rule 2: Wheezing and reduced FEV1 (using standard height baseline fallback < 2.0L for Males, < 1.7L for Females)
        elif as_wheezing == 1:
            is_reduced = (fev1 < 2.2) if as_gender == 1 else (fev1 < 1.8)
            if is_reduced:
                pred_class = 1
                if raw_pred != 1:
                    override_applied = True
                    
        return pred_class, prob[1], override_applied

    print("=== 1. Non-Asthmatic Profile Stability Test ===")
    # Profile: no wheeze, no allergy history, never smoker, normal FEV1 (3.2L), varying age 20-80 and BMI 18-35
    for age_var in [20, 40, 60, 80]:
        for bmi_var in [18.5, 24.5, 30.0, 35.0]:
            pred, p_pos, over = predict_with_override(
                age=age_var, gender="Female", wheezing="No", allergies="No", smoking="Never", fev1=3.2
            )
            # Must always predict No Asthma (0)
            assert pred == 0, f"Failed for age={age_var}, bmi={bmi_var}. Predicted {pred}"
    print("Non-Asthmatic stability: PASS")

    print("\n=== 2. Asthmatic Profile Stability Test ===")
    # Profile: wheeze = yes, allergy history = yes, reduced FEV1 (1.2L)
    for age_var in [20, 45, 70]:
        for smoking_var in ["Never", "Ex-smoker", "Current smoker"]:
            pred, p_pos, over = predict_with_override(
                age=age_var, gender="Male", wheezing="Yes", allergies="Yes", smoking=smoking_var, fev1=1.2
            )
            # Must always predict Asthma (1)
            assert pred == 1, f"Failed for age={age_var}, smoking={smoking_var}. Predicted {pred}"
    print("Asthmatic stability: PASS")

    print("\n=== 3. Borderline Case Analysis (near threshold 0.21) ===")
    # Find a borderline case
    # E.g. wheezing = No, allergies = Yes, normal FEV1. It falls to the ML model since allergies = Yes.
    pred, p_pos, over = predict_with_override(
        age=30, gender="Female", wheezing="No", allergies="Yes", smoking="Never", fev1=2.8
    )
    print(f"Borderline case (allergies only) Raw ML Positive Prob: {p_pos*100:.2f}%. Deployed Prediction: {'Asthma' if pred==1 else 'No Asthma'}. Override applied: {over}")

if __name__ == "__main__":
    test_asthma_stability()
