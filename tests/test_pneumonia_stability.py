import os
import joblib
import numpy as np

def test_pneumonia_stability():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    BUNDLE_PATH = os.path.join(BASE_DIR, "saved_models", "pneumonia_bundle.pkl")
    
    assert os.path.exists(BUNDLE_PATH), f"Bundle not found at {BUNDLE_PATH}"
    bundle = joblib.load(BUNDLE_PATH)
    model = bundle["best_model"]
    
    # Feature order: ['Fever', 'Cough', 'ChestPain', 'WBCCount', 'SpO2', 'RespiratorySymptom']
    # Mappings inside app.py:
    # Fever: No Fever/Low -> 1, Moderate -> 2, High -> 3
    # Cough: No Cough/Dry -> 1, Wet -> 2, Bloody -> 3
    # ChestPain: No Chest Pain/Mild -> 1, Moderate -> 2, Severe -> 3
    # RespiratorySymptom: None/Mild -> 1, Moderate -> 2, Severe -> 3
    
    def predict_pneu(fever, cough, chest_pain, wbc, spo2, resp_sym):
        # clean and encode
        f_val = 3 if fever == "High" else (2 if fever == "Moderate" else 1)
        c_val = 3 if cough == "Bloody" else (2 if cough == "Wet" else 1)
        cp_val = 3 if chest_pain == "Severe" else (2 if chest_pain == "Moderate" else 1)
        rs_val = 3 if resp_sym == "Severe" else (2 if resp_sym == "Moderate" else 1)
        
        features = np.array([[f_val, c_val, cp_val, wbc, spo2, rs_val]])
        prob = model.predict_proba(features)[0]
        pred_class = int(np.argmax(prob))
        return pred_class, prob
        
    print("=== 1. Asymptomatic / Healthy Baseline check ===")
    pred, prob = predict_pneu(
        fever="No Fever", cough="No Cough", chest_pain="No Chest Pain", wbc=7000, spo2=98, resp_sym="None"
    )
    print(f"Asymptomatic patient -> predicted class: {pred} (No Pneumonia: {prob[0]*100:.2f}%)")
    assert pred == 0, f"Expected 0 (No Pneumonia), got {pred}"
    print("Asymptomatic check: PASS")

    print("\n=== 2. Symptomatic Baseline check ===")
    pred, prob = predict_pneu(
        fever="High", cough="Bloody", chest_pain="Severe", wbc=12000, spo2=88, resp_sym="Severe"
    )
    print(f"Symptomatic patient -> predicted class: {pred} (Pneumonia: {prob[1]*100:.2f}%)")
    assert pred == 1, f"Expected 1 (Pneumonia), got {pred}"
    print("Symptomatic check: PASS")

if __name__ == "__main__":
    test_pneumonia_stability()
