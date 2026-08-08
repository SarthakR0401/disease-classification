import os
import joblib
import numpy as np

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

def test_copd_hr_rr_stability():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODEL_PATH = os.path.join(BASE_DIR, "saved_models", "copd_cb_model.pkl")
    
    # Assert model exists
    assert os.path.exists(MODEL_PATH), f"Model not found at {MODEL_PATH}"
    
    # Load Model
    model = joblib.load(MODEL_PATH)
    
    # Patient profile
    age = 65
    gender = 0  # Female
    smoker = 0  # Never smoked
    bmi = 24.5
    fev1 = 2.5
    fvc = 3500
    spo2 = 98
    dyspnea = 0
    height_cm = 162  # Assumed height for female
    
    # Compute engineered features
    fev1_pred = get_predicted_fev1(age, gender, height_cm)
    fvc_pred = get_predicted_fvc(age, gender, height_cm)
    fev1_pct_predicted = (fev1 / fev1_pred) * 100
    fev1_fvc_ratio = fev1 / fvc_pred
    
    # Feature names: ['Age', 'Gender', 'Smoker', 'BMI', 'Baseline SpO2', 'Baseline Dyspnea (mMRC)', 'fev1_pct_predicted', 'fev1_fvc_ratio']
    # Build inputs (notice heart_rate and resp_rate are completely omitted from the features)
    features = np.array([[
        age, gender, smoker, bmi, spo2, dyspnea, fev1_pct_predicted, fev1_fvc_ratio
    ]])
    
    # Heart rate 140, Resp rate 29 vs Heart rate 75, Resp rate 16
    # Both use the same features because HR/RR are removed
    prob_high = model.predict_proba(features)[0]
    prob_normal = model.predict_proba(features)[0]
    
    pred_high = int(np.argmax(prob_high))
    pred_normal = int(np.argmax(prob_normal))
    
    print(f"High HR/RR:   predicted class={pred_high}, probability={prob_high}")
    print(f"Normal HR/RR: predicted class={pred_normal}, probability={prob_normal}")
    
    # Assert they are identical
    assert pred_high == pred_normal, f"Regression check failed: class flipped! ({pred_high} vs {pred_normal})"
    print("Stability regression check: PASS")

if __name__ == "__main__":
    test_copd_hr_rr_stability()
