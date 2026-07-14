import sys
import os
import json

# Add parent directory to path to allow direct import of drm
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import drm

# Standardize output printing
sys.stdout.reconfigure(encoding='utf-8')

# --- Test Patients Cases ---

test_cases = [
    {
        "name": "Case 1: Standard COPD Profile (FEV1 & FVC provided)",
        "data": {
            "age": 65,
            "gender": "Male",
            "smoking_status": "Yes",
            "bmi": 28.5,
            "fev1": 1.1,                 # Trigger: FEV1 & FVC provided
            "fvc": 3.1,                  # Trigger: FEV1 & FVC provided
            "respiratory_rate": 22.0,
            "dyspnea_score": "mMRC 3",
            "heartrate": 78.0,
            "chest_xray_available": False,
            "hrct_available": False
        }
    },
    {
        "name": "Case 2: Standard IPF Profile (HRCT Imaging available)",
        "data": {
            "age": 70,
            "gender": "Female",
            "smoking_status": "ex-smoker",
            "fvc": 2.1,
            "dlco": 55.0,
            "weeks": 12,
            "hrct_available": True,       # Trigger: HRCT available
            "chest_xray_available": False
        }
    },
    {
        "name": "Case 3: Standard Pneumonia Profile (Chest X-ray & Fever present)",
        "data": {
            "age": 45,
            "gender": "Male",
            "temperature": 38.9,         # Trigger: Fever present
            "fever": "High",
            "chest_xray_available": True, # Trigger: Chest X-ray available
            "cough": "Wet",
            "chest_pain": "Moderate",
            "wbc_count": 12.5,
            "spo2": 91.0,
            "shortness_of_breath": "Mild"
        }
    },
    {
        "name": "Case 4: Standard Asthma Profile (Allergies, Wheezing & Peak Flow)",
        "data": {
            "age": 25,
            "gender": "Female",
            "allergy_history": "Yes",     # Trigger: Allergy History
            "wheezing": "Yes",            # Trigger: Wheezing
            "peak_flow": 320.0,           # Trigger: Peak Flow
            "fev1": 2.8,                  # Note: triggers COPD too? No, FVC is missing.
            "bmi": 21.0,
            "cough": "Dry",
            "shortness_of_breath": "Yes"
        }
    },
    {
        "name": "Case 5: Overlapping COPD & IPF Profile (Spirometry + HRCT + DLCO)",
        "data": {
            "age": 68,
            "gender": "Male",
            "smoking_status": "Ex-smoker",
            "bmi": 26.4,
            "fev1": 1.2,                 # Trigger: COPD (FEV1 & FVC)
            "fvc": 2.2,                  # Trigger: COPD & IPF FVC
            "dlco": 50.0,
            "hrct_available": True,       # Trigger: IPF (HRCT)
            "chest_xray_available": False
        }
    }
]

print("=" * 80)
print("              DISEASE ROUTING MODULE (DRM) - TEST SUITE")
print("=" * 80)

for idx, case in enumerate(test_cases, 1):
    print(f"\n--- Running {case['name']} ---")
    print("Patient Clinical Record:")
    for k, v in case['data'].items():
        print(f"  {k:<25}: {v}")
        
    # Execute DRM prediction
    result = drm.predict_patient(case['data'])
    
    print("\nRouting Results:")
    print(f"  Routed Model(s)                 : {result['routing']['selected_models']}")
    print(f"  Required Feature Set Identified  : {result['routing']['required_feature_set_identified']}")
    print(f"  Data Ready for Inference        : {result['routing']['data_ready_for_prediction']}")
    
    print("\nModel Prediction(s) & Probabilities:")
    if result["predictions"]:
        for model_name, pred_res in result["predictions"].items():
            print(f"  * {model_name} Model Prediction:")
            if pred_res.get("status") == "Error":
                print(f"    ❌ Error: {pred_res.get('message')}")
            else:
                print(f"    - Predicted Class   : {pred_res.get('prediction_class')}")
                print(f"    - Confidence Score  : {pred_res.get('confidence_score')*100:.2f}%")
                if "probability_disease" in pred_res:
                    print(f"    - Disease Prob      : {pred_res.get('probability_disease')*100:.2f}%")
                if "stage_probabilities" in pred_res:
                    print("    - Stage Probabilities:")
                    for stage, p in pred_res.get("stage_probabilities").items():
                        print(f"      * {stage}: {p*100:.2f}%")
                if "severity_probabilities" in pred_res:
                    print("    - Severity Probabilities:")
                    for severity, p in pred_res.get("severity_probabilities").items():
                        print(f"      * {severity}: {p*100:.2f}%")
                print(f"    - Explanation       : {pred_res.get('explanation')}")
    else:
        print(f"  ⚠️ Status: {result.get('status')}")
    print("-" * 80)
