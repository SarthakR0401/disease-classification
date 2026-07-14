import os
import joblib
import pandas as pd
import numpy as np

# Directory where the trained models are saved
MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "IPYNB"))

# --- Helper Functions for Feature Encoding & Standarization ---

def to_binary(val):
    """Maps yes/no, true/false, 1/0 to 1/0."""
    if val is None:
        return None
    val_str = str(val).strip().lower()
    if val_str in ['1', '1.0', 'true', 'yes', 'y', 't', 'active']:
        return 1
    return 0

def to_gender_binary_asthma(val):
    """Asthma model encoding: Female=1, Male=0."""
    if val is None:
        return 0
    val_str = str(val).strip().lower()
    if val_str in ['female', 'f', '1']:
        return 1
    return 0

def to_gender_binary_copd(val):
    """COPD model encoding: Male=1, Female=0."""
    if val is None:
        return 0
    val_str = str(val).strip().lower()
    if val_str in ['male', 'm', '1']:
        return 1
    return 0

def to_smoker_binary_copd(val):
    """COPD model encoding: Yes=1, No=0."""
    return to_binary(val)

def extract_mmrc_score(val):
    """Extracts numeric digit from Baseline Dyspnea (mMRC)."""
    if val is None:
        return 3.0  # COPD median default
    # Extract number
    import re
    match = re.search(r'(\d+)', str(val))
    if match:
        return float(match.group(1))
    return 3.0

def standardize_ipf_category(val, classes):
    """Standardizes string values to match IPF label encoder classes dynamically."""
    if val is None:
        return classes[0]
    val_str = str(val).strip().lower()
    # Check for direct match or first letter prefix match
    for c in classes:
        c_lower = c.lower()
        if val_str == c_lower or val_str in c_lower or c_lower in val_str:
            return c
        if val_str[0] == c_lower[0]:
            return c
    return classes[0]

# --- Core Disease Routing Dispatcher ---

def route_patient(patient_data):
    """
    Evaluates patient parameters to dispatch to the correct model(s).
    
    Rules:
    - Route to COPD: If both FEV1 and FVC are present (spirometry parameters).
    - Route to IPF: If HRCT is available OR if FVC and DLCO are present.
    - Route to Pneumonia: If Chest X-ray is available AND Fever is present.
    - Route to Asthma: If Allergy History, Wheezing, and Peak Flow are all available.
    """
    selected_models = []
    
    # 1. COPD Rule: spirometry (FEV1 & FVC)
    if patient_data.get("fev1") is not None and patient_data.get("fvc") is not None:
        selected_models.append("COPD")
        
    # 2. IPF Rule: HRCT imaging OR (FVC + DLCO)
    has_hrct = patient_data.get("hrct_available") is True
    has_ipf_pulmonary = patient_data.get("fvc") is not None and patient_data.get("dlco") is not None
    if has_hrct or has_ipf_pulmonary:
        selected_models.append("IPF")
        
    # 3. Pneumonia Rule: Chest X-ray + Fever
    has_xray = patient_data.get("chest_xray_available") is True
    has_fever = (
        patient_data.get("fever") in ["Low", "Moderate", "High", "Yes", True, 1] or 
        (patient_data.get("temperature") is not None and patient_data.get("temperature") > 37.5)
    )
    if has_xray and has_fever:
        selected_models.append("Pneumonia")
        
    # 4. Asthma Rule: Allergy History + Wheezing + Peak Flow
    has_allergy = to_binary(patient_data.get("allergy_history")) == 1
    has_wheezing = to_binary(patient_data.get("wheezing")) == 1
    has_peak_flow = patient_data.get("peak_flow") is not None
    if has_allergy and has_wheezing and has_peak_flow:
        selected_models.append("Asthma")
        
    return {
        "selected_models": selected_models,
        "required_feature_set_identified": "Yes" if len(selected_models) > 0 else "No",
        "data_ready_for_prediction": "Yes" if len(selected_models) > 0 else "No"
    }

# --- Explanation Generator (SHAP Fallback) ---

def explain_prediction(model, features, X_input, best_model_name):
    """
    Generates a local explanation highlighting feature contributions.
    Uses tree importances or model coefficients scaled by standard metrics.
    """
    importances = None
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = model.coef_[0]
        
    if importances is None:
        return {
            "explanation": "No direct feature importance explanation available for this model type.",
            "feature_contributions": {}
        }
        
    contribs = {}
    for feat, imp, val in zip(features, importances, X_input[0]):
        contribs[feat] = {
            "importance": float(imp),
            "value": float(val) if isinstance(val, (int, float)) else str(val)
        }
        
    sorted_features = sorted(contribs.items(), key=lambda x: abs(x[1]["importance"]), reverse=True)
    
    explanation_text = f"Feature contributions ({best_model_name}): "
    elements = []
    for feat, detail in sorted_features[:3]:
        elements.append(f"{feat} (value: {detail['value']}, contribution: {detail['importance']*100:.1f}%)")
    explanation_text += ", and ".join(elements) + "."
    
    return {
        "feature_contributions": contribs,
        "explanation": explanation_text
    }

# --- Core Inference Engine ---

def predict_patient(patient_data):
    """
    Standardized inference entrypoint.
    Runs the patient record through routing and executes all activated models.
    """
    routing_result = route_patient(patient_data)
    active_models = routing_result["selected_models"]
    
    predictions = {}
    
    if not active_models:
        return {
            "routing": routing_result,
            "predictions": {},
            "status": "No model routed. Please supply relevant clinical parameters (spirometry, HRCT, X-ray + Fever, or allergy/peak flow)."
        }
        
    for model_name in active_models:
        try:
            if model_name == "Asthma":
                predictions["Asthma"] = _predict_asthma(patient_data)
            elif model_name == "COPD":
                predictions["COPD"] = _predict_copd(patient_data)
            elif model_name == "IPF":
                predictions["IPF"] = _predict_ipf(patient_data)
            elif model_name == "Pneumonia":
                predictions["Pneumonia"] = _predict_pneumonia(patient_data)
        except FileNotFoundError as fnf:
            predictions[model_name] = {
                "status": "Error",
                "message": f"Trained model files are missing for {model_name}. Please run 'python src/train_all.py' to train and save them."
            }
        except Exception as e:
            predictions[model_name] = {
                "status": "Error",
                "message": f"Inference execution failed: {str(e)}"
            }
            
    return {
        "routing": routing_result,
        "predictions": predictions,
        "status": "Success"
    }

# --- Private Model Executors ---

def _predict_asthma(patient):
    model_path = os.path.join(MODELS_DIR, "best_asthma_clinical_model.pkl")
    scaler_path = os.path.join(MODELS_DIR, "asthma_scaler.pkl")
    features_path = os.path.join(MODELS_DIR, "asthma_feature_columns.pkl")
    
    if not (os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(features_path)):
        raise FileNotFoundError("Asthma model files missing")
        
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    features = joblib.load(features_path)
    
    # Map patient parameters to Asthma schema
    # Features in order: ['Age', 'Gender', 'Ethnicity', 'EducationLevel', 'BMI', 'Smoking', 'PhysicalActivity', 'DietQuality', 'SleepQuality', 'PollutionExposure', 'PollenExposure', 'DustExposure', 'PetAllergy', 'FamilyHistoryAsthma', 'HistoryOfAllergies', 'Eczema', 'HayFever', 'GastroesophagealReflux', 'LungFunctionFEV1', 'LungFunctionFVC', 'Wheezing', 'ShortnessOfBreath', 'ChestTightness', 'Coughing', 'NighttimeSymptoms', 'ExerciseInduced']
    
    data_dict = {
        'Age': patient.get("age", 40.0),
        'Gender': to_gender_binary_asthma(patient.get("gender")),
        'Ethnicity': int(patient.get("ethnicity", 0)),
        'EducationLevel': int(patient.get("education_level", 1)),
        'BMI': patient.get("bmi", 22.0),
        'Smoking': to_binary(patient.get("smoking_status")),
        'PhysicalActivity': float(patient.get("physical_activity", 5.0)),
        'DietQuality': float(patient.get("diet_quality", 5.0)),
        'SleepQuality': float(patient.get("sleep_quality", 6.0)),
        'PollutionExposure': float(patient.get("pollution_exposure", 3.0)),
        'PollenExposure': float(patient.get("pollen_exposure", 3.0)),
        'DustExposure': float(patient.get("dust_exposure", 3.0)),
        'PetAllergy': to_binary(patient.get("pet_allergy")),
        'FamilyHistoryAsthma': to_binary(patient.get("family_history_asthma")),
        'HistoryOfAllergies': to_binary(patient.get("allergy_history")),
        'Eczema': to_binary(patient.get("eczema")),
        'HayFever': to_binary(patient.get("hay_fever")),
        'GastroesophagealReflux': to_binary(patient.get("gastroesophageal_reflux")),
        'LungFunctionFEV1': patient.get("fev1", 2.5),
        'LungFunctionFVC': patient.get("fvc", 3.5),
        'Wheezing': to_binary(patient.get("wheezing")),
        'ShortnessOfBreath': to_binary(patient.get("shortness_of_breath")),
        'ChestTightness': to_binary(patient.get("chest_tightness", patient.get("chest_pain"))),
        'Coughing': to_binary(patient.get("cough")),
        'NighttimeSymptoms': to_binary(patient.get("nighttime_symptoms")),
        'ExerciseInduced': to_binary(patient.get("exercise_induced"))
    }
    
    # Ensure columns match asthma feature schema order
    X_df = pd.DataFrame([data_dict])[features]
    X_arr = X_df.values
    
    # Scale if Logistic Regression
    is_lr = type(model).__name__ in ["LogisticRegression", "SVC"]
    if is_lr:
        X_eval = pd.DataFrame(scaler.transform(X_df), columns=features)
    else:
        X_eval = X_df
        
    pred = model.predict(X_eval)[0]
    prob = model.predict_proba(X_eval)[0]
    
    class_label = "Asthma" if pred == 1 else "Healthy (No Asthma)"
    confidence = float(prob[1] if pred == 1 else prob[0])
    
    explanations = explain_prediction(model, features, X_arr, type(model).__name__)
    
    return {
        "prediction_class": class_label,
        "probability_disease": float(prob[1]),
        "confidence_score": confidence,
        "explanation": explanations["explanation"],
        "feature_contributions": explanations["feature_contributions"]
    }

def _predict_copd(patient):
    model_path = os.path.join(MODELS_DIR, "best_copd_clinical_model.pkl")
    scaler_path = os.path.join(MODELS_DIR, "copd_scaler.pkl")
    features_path = os.path.join(MODELS_DIR, "copd_feature_columns.pkl")
    
    if not (os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(features_path)):
        raise FileNotFoundError("COPD model files missing")
        
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    features = joblib.load(features_path)
    
    # Imputation values calculated from dataset
    copd_medians = {
        'Age': 64.0, 'Gender': 0, 'Smoker': 0, 'BMI': 30.52,
        'Baseline FEV1': 1.33, 'Baseline SpO2': 94.83,
        'Baseline Respiration Rate': 28.09, 'Baseline Dyspnea (mMRC)': 3.0,
        'Baseline heartrate': 71.0
    }
    
    data_dict = {
        'Age': float(patient.get("age", copd_medians['Age'])),
        'Gender': to_gender_binary_copd(patient.get("gender")),
        'Smoker': to_smoker_binary_copd(patient.get("smoking_status")),
        'BMI': float(patient.get("bmi", copd_medians['BMI'])),
        'Baseline FEV1': float(patient.get("fev1", copd_medians['Baseline FEV1'])),
        'Baseline SpO2': float(patient.get("spo2", copd_medians['Baseline SpO2'])),
        'Baseline Respiration Rate': float(patient.get("respiratory_rate", copd_medians['Baseline Respiration Rate'])),
        'Baseline Dyspnea (mMRC)': extract_mmrc_score(patient.get("dyspnea_score", patient.get("shortness_of_breath"))),
        'Baseline heartrate': float(patient.get("heartrate", copd_medians['Baseline heartrate']))
    }
    
    X_df = pd.DataFrame([data_dict])[features]
    X_arr = X_df.values
    
    is_scaled_model = type(model).__name__ in ["LogisticRegression", "SVC"]
    if is_scaled_model:
        X_eval = pd.DataFrame(scaler.transform(X_df), columns=features)
    else:
        X_eval = X_df
        
    pred = int(model.predict(X_eval)[0])
    prob = model.predict_proba(X_eval)[0]
    
    stage_names = ['Stage 1 (Mild)', 'Stage 2 (Moderate)', 'Stage 3 (Severe)', 'Stage 4 (Very Severe)']
    class_label = stage_names[pred]
    confidence = float(prob[pred])
    
    explanations = explain_prediction(model, features, X_arr, type(model).__name__)
    
    return {
        "prediction_class": class_label,
        "stage_probabilities": {stage_names[i]: float(prob[i]) for i in range(len(stage_names))},
        "confidence_score": confidence,
        "explanation": explanations["explanation"],
        "feature_contributions": explanations["feature_contributions"]
    }

def _predict_ipf(patient):
    model_path = os.path.join(MODELS_DIR, "best_ipf_clinical_model.pkl")
    scaler_path = os.path.join(MODELS_DIR, "ipf_scaler.pkl")
    encoders_path = os.path.join(MODELS_DIR, "ipf_label_encoders.pkl")
    
    if not (os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(encoders_path)):
        raise FileNotFoundError("IPF model files missing")
        
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    encoders = joblib.load(encoders_path)
    
    features = ['Weeks', 'FVC', 'Percent', 'Age', 'Sex', 'SmokingStatus']
    
    # Calculate estimated FVC percent if missing
    fvc_val = patient.get("fvc", 2000.0)
    percent_val = patient.get("percent")
    if percent_val is None:
        # Standard clinical prediction estimate
        percent_val = min(100.0, max(20.0, (fvc_val / 3000.0) * 100.0))
        
    # Standardize categoricals
    sex_std = standardize_ipf_category(patient.get("gender"), encoders["Sex"].classes_)
    smoke_std = standardize_ipf_category(patient.get("smoking_status"), encoders["SmokingStatus"].classes_)
    
    # Transform using encoders
    sex_encoded = encoders["Sex"].transform([sex_std])[0]
    smoke_encoded = encoders["SmokingStatus"].transform([smoke_std])[0]
    
    data_dict = {
        'Weeks': int(patient.get("weeks", 0)),
        'FVC': float(fvc_val),
        'Percent': float(percent_val),
        'Age': float(patient.get("age", 65.0)),
        'Sex': int(sex_encoded),
        'SmokingStatus': int(smoke_encoded)
    }
    
    X_df = pd.DataFrame([data_dict])[features]
    X_arr = X_df.values
    
    is_lr = type(model).__name__ in ["LogisticRegression", "SVC"]
    if is_lr:
        X_eval = pd.DataFrame(scaler.transform(X_df), columns=features)
    else:
        X_eval = X_df
        
    pred = int(model.predict(X_eval)[0])
    prob = model.predict_proba(X_eval)[0]
    
    severity_classes = encoders["IPF_Severity"].classes_
    class_label = severity_classes[pred]
    confidence = float(prob[pred])
    
    explanations = explain_prediction(model, features, X_arr, type(model).__name__)
    
    # Re-map numerical keys of encoders to feature names for explanation visualization
    decoded_contributions = {}
    for k, v in explanations["feature_contributions"].items():
        if k == 'Sex':
            v['value'] = sex_std
        elif k == 'SmokingStatus':
            v['value'] = smoke_std
        decoded_contributions[k] = v
        
    return {
        "prediction_class": class_label,
        "severity_probabilities": {severity_classes[i]: float(prob[i]) for i in range(len(severity_classes))},
        "confidence_score": confidence,
        "explanation": explanations["explanation"].replace("Sex (value: 0", f"Sex (value: {sex_std}").replace("Sex (value: 1", f"Sex (value: {sex_std}"),
        "feature_contributions": decoded_contributions
    }

def _predict_pneumonia(patient):
    model_path = os.path.join(MODELS_DIR, "best_pneumonia_clinical_model.pkl")
    scaler_path = os.path.join(MODELS_DIR, "pneumonia_scaler.pkl")
    features_path = os.path.join(MODELS_DIR, "pneumonia_feature_columns.pkl")
    imputations_path = os.path.join(MODELS_DIR, "pneumonia_imputation_values.pkl")
    
    if not (os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(features_path) and os.path.exists(imputations_path)):
        raise FileNotFoundError("Pneumonia model files missing")
        
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    features = joblib.load(features_path)
    imputations = joblib.load(imputations_path)
    
    # Feature maps
    fever_map = {"low": 1, "low-grade": 1, "moderate": 2, "high": 3, "yes": 2, "no": 0}
    cough_map = {"dry": 1, "wet": 2, "bloody": 3, "yes": 2, "no": 0}
    chest_map = {"mild": 1, "moderate": 2, "severe": 3, "yes": 2, "no": 0}
    resp_map = {"mild": 1, "moderate": 2, "severe": 3, "yes": 2, "no": 0}
    
    def map_cat(val, mapping, col_name):
        if val is None:
            return imputations.get(col_name, 1.0)
        if isinstance(val, (int, float)):
            return float(val)
        val_str = str(val).strip().lower()
        return float(mapping.get(val_str, imputations.get(col_name, 1.0)))
        
    fever_encoded = map_cat(patient.get("fever"), fever_map, "Fever")
    # If temperature is provided, update fever score
    temp = patient.get("temperature")
    if temp is not None and isinstance(temp, (int, float)):
        if temp < 37.3:
            fever_encoded = 0.0
        elif temp < 38.3:
            fever_encoded = 1.0
        elif temp < 39.3:
            fever_encoded = 2.0
        else:
            fever_encoded = 3.0
            
    cough_encoded = map_cat(patient.get("cough"), cough_map, "Cough")
    chest_encoded = map_cat(patient.get("chest_pain"), chest_map, "ChestPain")
    resp_encoded = map_cat(patient.get("shortness_of_breath"), resp_map, "RespiratorySymptom")
    
    data_dict = {
        'Fever': fever_encoded,
        'Cough': cough_encoded,
        'ChestPain': chest_encoded,
        'WBCCount': float(patient.get("wbc_count", imputations.get("WBCCount", 9.0))),
        'SpO2': float(patient.get("spo2", imputations.get("SpO2", 95.0))),
        'RespiratorySymptom': resp_encoded
    }
    
    X_df = pd.DataFrame([data_dict])[features]
    X_arr = X_df.values
    
    # Pneumonia bundle config
    is_scaled = type(model).__name__ in ["LogisticRegression", "SVC"]
    if is_scaled:
        X_eval = pd.DataFrame(scaler.transform(X_df), columns=features)
    else:
        X_eval = X_df
        
    pred = model.predict(X_eval)[0]
    prob = model.predict_proba(X_eval)[0]
    
    class_label = "Pneumonia" if pred == 1 else "No Pneumonia"
    confidence = float(prob[1] if pred == 1 else prob[0])
    
    explanations = explain_prediction(model, features, X_arr, type(model).__name__)
    
    # Map numeric representations back to strings for clarity
    rev_maps = {
        'Fever': {1.0: 'Low', 2.0: 'Moderate', 3.0: 'High', 0.0: 'None'},
        'Cough': {1.0: 'Dry', 2.0: 'Wet', 3.0: 'Bloody', 0.0: 'None'},
        'ChestPain': {1.0: 'Mild', 2.0: 'Moderate', 3.0: 'Severe', 0.0: 'None'},
        'RespiratorySymptom': {1.0: 'Mild', 2.0: 'Moderate', 3.0: 'Severe', 0.0: 'None'}
    }
    
    decoded_contribs = {}
    for k, v in explanations["feature_contributions"].items():
        if k in rev_maps and v['value'] in rev_maps[k]:
            v['value'] = rev_maps[k][v['value']]
        decoded_contribs[k] = v
        
    return {
        "prediction_class": class_label,
        "probability_disease": float(prob[1]),
        "confidence_score": confidence,
        "explanation": explanations["explanation"],
        "feature_contributions": decoded_contribs
    }
