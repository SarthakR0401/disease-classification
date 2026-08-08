import os
import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template

app = Flask(__name__, template_folder="templates", static_folder="static")

# --- Load Models & Utilities ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "saved_models")

# 1. Asthma
asthma_model = joblib.load(os.path.join(MODELS_DIR, "asthma_rf_model.pkl"))

# 2. IPF
ipf_model = joblib.load(os.path.join(MODELS_DIR, "ipf_lr_model.pkl"))
ipf_scaler = joblib.load(os.path.join(MODELS_DIR, "ipf_scaler.pkl"))
ipf_label_encoders = joblib.load(os.path.join(MODELS_DIR, "ipf_label_encoders.pkl"))

# 3. Pneumonia
pneu_bundle = joblib.load(os.path.join(MODELS_DIR, "pneumonia_bundle.pkl"))
pneu_model = pneu_bundle["best_model"]
pneu_scaler = pneu_bundle["scaler"]
pneu_imputer_vals = pneu_bundle["imputation_values"]

# 4. COPD
copd_model = joblib.load(os.path.join(MODELS_DIR, "copd_cb_model.pkl"))


def clean_float(val):
    if val is None or val == "":
        return None
    try:
        return float(val)
    except:
        return None
def get_predicted_fev1(age, gender, height_cm, race="Unknown"):
    is_male = (gender == 1 or gender == "Male")
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
    is_male = (gender == 1 or gender == "Male")
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


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/route_and_predict", methods=["POST"])
def route_and_predict():
    data = request.json or {}
    
    # Extract raw inputs
    age = clean_float(data.get("age"))
    gender = data.get("gender") # "Male" or "Female"
    smoking = data.get("smoking") # "Never smoked", "Ex-smoker", "Currently smokes"
    bmi = clean_float(data.get("bmi"))
    
    wheezing = data.get("wheezing") # "Yes" or "No"
    allergies = data.get("allergies") # "Yes" or "No"
    fev1 = clean_float(data.get("fev1"))
    
    fvc = clean_float(data.get("fvc"))
    fvc_percent = clean_float(data.get("fvc_percent"))
    
    fever = data.get("fever") # "Low", "Moderate", "High", "No Fever"
    cough = data.get("cough") # "Dry", "Wet", "Bloody", "No Cough"
    chest_pain = data.get("chest_pain") # "Mild", "Moderate", "Severe", "No Chest Pain"
    wbc_count = clean_float(data.get("wbc_count"))
    spo2 = clean_float(data.get("spo2"))
    resp_symptom = data.get("resp_symptom") # "Mild", "Moderate", "Severe", "None"
    
    resp_rate = clean_float(data.get("resp_rate"))
    dyspnea = data.get("dyspnea") # "mMRC0" to "mMRC5" or numeric 0-5
    heart_rate = clean_float(data.get("heart_rate"))
    
    # --- DRM Routing Logic ---
    active_models = {}
    predictions = {}
    
    # 1. COPD Routing check
    height_cm = clean_float(data.get("height_cm"))
    race = data.get("race", "Unknown")
    copd_req = [age, gender, smoking, bmi, height_cm, fev1, spo2, resp_rate, dyspnea, heart_rate]
    if all(x is not None for x in copd_req):
        active_models["COPD"] = "Active"
        try:
            copd_gender = 1 if gender == "Male" else 0
            copd_smoker = 1 if smoking in ["Currently smokes", "Ex-smoker"] else 0
            
            import re
            m = re.search(r'\d+', str(dyspnea))
            copd_dyspnea = int(m.group(0)) if m else 0
            
            # Feature engineering (NHANES III Reference Equations)
            fev1_pred = get_predicted_fev1(age, gender, height_cm, race)
            fvc_pred = get_predicted_fvc(age, gender, height_cm, race)
            
            fev1_pct_predicted = (fev1 / fev1_pred) * 100
            fev1_fvc_ratio = fev1 / fvc_pred
            
            features = np.array([[
                age, copd_gender, copd_smoker, bmi, spo2, copd_dyspnea, fev1_pct_predicted, fev1_fvc_ratio
            ]])
            
            prob = copd_model.predict_proba(features)[0]
            raw_pred_class = int(np.argmax(prob))
            stages = ["Stage 1 (No significant obstruction)", "Stage 2", "Stage 3", "Stage 4"]
            
            pred_class = raw_pred_class
            override_applied = False
            
            # GOLD Spirometric Staging boundaries
            if fev1_fvc_ratio >= 0.70:
                pred_class = 0  # Stage 1 / No obstruction
                if raw_pred_class != 0:
                    override_applied = True
            else:
                # Patient has clinical obstruction. Staging is strictly defined by FEV1% predicted.
                if fev1_pct_predicted >= 80.0:
                    expected_class = 0  # Stage 1
                elif fev1_pct_predicted >= 50.0:
                    expected_class = 1  # Stage 2
                elif fev1_pct_predicted >= 30.0:
                    expected_class = 2  # Stage 3
                else:
                    expected_class = 3  # Stage 4
                    
                if pred_class != expected_class:
                    pred_class = expected_class
                    override_applied = True
            
            confidence = 1.0 if override_applied else float(prob[pred_class])
            
            predictions["COPD"] = {
                "class": stages[pred_class],
                "probability": [float(p) for p in prob],
                "confidence": confidence,
                "stages": stages,
                "override_applied": override_applied,
                "raw_class": "Stage " + str(raw_pred_class + 1),
                "raw_confidence": float(prob[raw_pred_class]),
                "status": "Success"
            }
        except Exception as e:
            predictions["COPD"] = {"status": "Error", "message": str(e)}
    else:
        missing = []
        if age is None: missing.append("Age")
        if gender is None: missing.append("Gender")
        if smoking is None: missing.append("Smoking Status")
        if bmi is None: missing.append("BMI")
        if height_cm is None: missing.append("Height (cm)")
        if fev1 is None: missing.append("FEV1 (Spirometry)")
        if spo2 is None: missing.append("SpO2")
        if resp_rate is None: missing.append("Respiration Rate")
        if dyspnea is None: missing.append("Dyspnea Level (mMRC)")
        if heart_rate is None: missing.append("Heart Rate")
        active_models["COPD"] = {"status": "Inactive", "missing": missing}

    # 2. Asthma Routing check
    asthma_req = [age, gender, wheezing, allergies, smoking, fev1]
    if all(x is not None for x in asthma_req):
        active_models["Asthma"] = "Active"
        try:
            as_gender = 1 if gender == "Male" else 0
            as_wheezing = 1 if wheezing == "Yes" else 0
            as_allergies = 1 if allergies == "Yes" else 0
            as_smoking = 1 if smoking == "Currently smokes" else 0
            
            features = np.array([[
                age, as_gender, as_wheezing, as_allergies, as_smoking, fev1
            ]])
            
            prob = asthma_model.predict_proba(features)[0]
            
            # Calculate FEV1% predicted if height is available, otherwise fallback to absolute cuts
            fev1_pred = get_predicted_fev1(age, gender, height_cm, race) if height_cm is not None else None
            fev1_pct = (fev1 / fev1_pred) * 100 if fev1_pred is not None else None
            
            # Determine if FEV1 is clinically reduced (<80% predicted or absolute cutoffs)
            is_reduced = False
            if fev1_pct is not None:
                is_reduced = fev1_pct < 80.0
            else:
                is_reduced = (fev1 < 2.2) if (gender == "Male" or gender == 1) else (fev1 < 1.8)
            
            # Clinical Asthma Staging Decision Tree (100% GINA rule-based)
            raw_pred_class = 1 if prob[1] >= 0.21 else 0
            
            # 100% Deterministic GINA Rule Tree (GINA 2023, Figure 1-2)
            # A history of variable respiratory symptoms (wheezing) is a prerequisite.
            if as_wheezing == 1 and as_allergies == 1:
                pred_class = 1  # Wheezing + Allergies -> Asthma
            elif as_wheezing == 1 and is_reduced:
                pred_class = 1  # Wheezing + Airflow obstruction -> Asthma
            else:
                pred_class = 0  # Low Risk / No typical respiratory symptoms for asthma
                
            override_applied = True
            confidence = 1.0
            
            predictions["Asthma"] = {
                "class": "1 (Asthma Detected)" if pred_class == 1 else "0 (No Asthma)",
                "probability": [float(p) for p in prob],
                "confidence": confidence,
                "stages": ["0 (No Asthma)", "1 (Asthma)"],
                "override_applied": override_applied,
                "raw_class": "1 (Asthma Detected)" if raw_pred_class == 1 else "0 (No Asthma)",
                "raw_confidence": float(prob[raw_pred_class]),
                "status": "Success"
            }
        except Exception as e:
            predictions["Asthma"] = {"status": "Error", "message": str(e)}
    else:
        missing = []
        if age is None: missing.append("Age")
        if gender is None: missing.append("Gender")
        if wheezing is None: missing.append("Wheezing Symptom")
        if allergies is None: missing.append("Allergy History")
        if smoking is None: missing.append("Smoking Status")
        if fev1 is None: missing.append("FEV1 (Spirometry)")
        active_models["Asthma"] = {"status": "Inactive", "missing": missing}

    # 3. IPF Routing check
    ipf_req = [fvc, fvc_percent, age, gender, smoking]
    if all(x is not None for x in ipf_req):
        active_models["IPF"] = "Active"
        try:
            ipf_sex = 1 if gender == "Male" else 0
            ipf_smoke = 2
            if smoking == "Currently smokes":
                ipf_smoke = 0
            elif smoking == "Ex-smoker":
                ipf_smoke = 1
                
            features_raw = np.array([[
                fvc, fvc_percent, age, ipf_sex, ipf_smoke
            ]])
            
            features_scaled = ipf_scaler.transform(features_raw)
            prob = ipf_model.predict_proba(features_scaled)[0]
            pred_class = int(np.argmax(prob))
            stages = ["Low Risk (Mild)", "High Risk (Moderate/Severe)"]
            confidence = float(prob[pred_class])
            
            predictions["IPF"] = {
                "class": stages[pred_class],
                "probability": [float(p) for p in prob],
                "confidence": confidence,
                "stages": stages,
                "screening_risk_score": float(prob[1]),
                "status": "Success"
            }
        except Exception as e:
            predictions["IPF"] = {"status": "Error", "message": str(e)}
    else:
        missing = []
        if fvc is None: missing.append("FVC (mL)")
        if fvc_percent is None: missing.append("FVC Predicted %")
        if age is None: missing.append("Age")
        if gender is None: missing.append("Gender")
        if smoking is None: missing.append("Smoking Status")
        active_models["IPF"] = {"status": "Inactive", "missing": missing}

    # 4. Pneumonia Routing check
    pneu_req = [fever, cough, chest_pain, spo2, resp_symptom]
    if all(x is not None for x in pneu_req):
        active_models["Pneumonia"] = "Active"
        try:
            fever_map = {"No Fever": 1, "Low": 1, "Moderate": 2, "High": 3}
            cough_map = {"No Cough": 1, "Dry": 1, "Wet": 2, "Bloody": 3}
            cp_map = {"No Chest Pain": 1, "Mild": 1, "Moderate": 2, "Severe": 3}
            rs_map = {"None": 1, "Mild": 1, "Moderate": 2, "Severe": 3}
            
            pneu_fever = fever_map.get(fever, 1)
            pneu_cough = cough_map.get(cough, 1)
            pneu_cp = cp_map.get(chest_pain, 1)
            pneu_rs = rs_map.get(resp_symptom, 1)
            pneu_wbc = wbc_count if wbc_count is not None else pneu_imputer_vals["WBCCount"]
            
            features_raw = np.array([[
                pneu_fever, pneu_cough, pneu_cp, pneu_wbc, spo2, pneu_rs
            ]])
            
            
            prob = pneu_model.predict_proba(features_raw)[0]
            pred_class = int(np.argmax(prob))
            confidence = float(prob[pred_class])
            
            predictions["Pneumonia"] = {
                "class": "1 (Pneumonia Detected)" if pred_class == 1 else "0 (No Pneumonia)",
                "probability": [float(p) for p in prob],
                "confidence": confidence,
                "stages": ["0 (No Pneumonia)", "1 (Pneumonia)"],
                "status": "Success"
            }
        except Exception as e:
            predictions["Pneumonia"] = {"status": "Error", "message": str(e)}
    else:
        missing = []
        if fever is None: missing.append("Fever")
        if cough is None: missing.append("Cough Status")
        if chest_pain is None: missing.append("Chest Pain")
        if spo2 is None: missing.append("SpO2")
        if resp_symptom is None: missing.append("Shortness of breath")
        active_models["Pneumonia"] = {"status": "Inactive", "missing": missing}
        
    return jsonify({
        "active_models": active_models,
        "predictions": predictions
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
