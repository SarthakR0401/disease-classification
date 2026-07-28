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
    copd_req = [age, gender, smoking, bmi, fev1, spo2, resp_rate, dyspnea, heart_rate]
    if all(x is not None for x in copd_req):
        active_models["COPD"] = "Active"
        try:
            copd_gender = 1 if gender == "Male" else 0
            copd_smoker = 1 if smoking in ["Currently smokes", "Ex-smoker"] else 0
            
            import re
            m = re.search(r'\d+', str(dyspnea))
            copd_dyspnea = int(m.group(0)) if m else 0
            
            features = np.array([[
                age, copd_gender, copd_smoker, bmi, fev1, spo2, resp_rate, copd_dyspnea, heart_rate
            ]])
            
            prob = copd_model.predict_proba(features)[0]
            pred_class = int(np.argmax(prob))
            stages = ["Stage 1", "Stage 2", "Stage 3", "Stage 4"]
            confidence = float(prob[pred_class])
            
            predictions["COPD"] = {
                "class": stages[pred_class],
                "probability": [float(p) for p in prob],
                "confidence": confidence,
                "stages": stages,
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
            pred_class = 1 if prob[1] >= 0.30 else 0
            confidence = float(prob[pred_class])
            
            predictions["Asthma"] = {
                "class": "1 (Asthma Detected)" if pred_class == 1 else "0 (No Asthma)",
                "probability": [float(p) for p in prob],
                "confidence": confidence,
                "stages": ["0 (No Asthma)", "1 (Asthma)"],
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
            stages = ["Mild", "Moderate", "Severe"]
            confidence = float(prob[pred_class])
            
            predictions["IPF"] = {
                "class": stages[pred_class],
                "probability": [float(p) for p in prob],
                "confidence": confidence,
                "stages": stages,
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
            
            features_scaled = pneu_scaler.transform(features_raw)
            prob = pneu_model.predict_proba(features_scaled)[0]
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
