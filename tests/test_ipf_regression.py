import os
import joblib
import numpy as np

def test_ipf_regression():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODEL_PATH = os.path.join(BASE_DIR, "saved_models", "ipf_lr_model.pkl")
    SCALER_PATH = os.path.join(BASE_DIR, "saved_models", "ipf_scaler.pkl")
    
    assert os.path.exists(MODEL_PATH), "IPF model not found"
    assert os.path.exists(SCALER_PATH), "IPF scaler not found"
    
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    
    # Profile 1: Healthy/Mild (FVC = 3500 mL, Percent = 85.0%, Age = 65, Female, Never smoked)
    # Sex: Male=1, Female=0. Smoker: Currently=0, Ex=1, Never=2.
    features_mild = np.array([[3500, 85.0, 65, 0, 2]])
    features_mild_scaled = scaler.transform(features_mild)
    prob_mild = model.predict_proba(features_mild_scaled)[0]
    pred_mild = int(np.argmax(prob_mild))
    
    # Profile 2: Impaired/Severe (FVC = 1200 mL, Percent = 40.0%, Age = 70, Male, Ex-smoker)
    features_severe = np.array([[1200, 40.0, 70, 1, 1]])
    features_severe_scaled = scaler.transform(features_severe)
    prob_severe = model.predict_proba(features_severe_scaled)[0]
    pred_severe = int(np.argmax(prob_severe))
    
    print(f"Profile 1 (Mild) predicted class: {pred_mild}, Probabilities: {prob_mild}")
    print(f"Profile 2 (Severe) predicted class: {pred_severe}, Probabilities: {prob_severe}")
    
    assert pred_mild == 0, f"Expected Mild (0) for Profile 1, got {pred_mild}"
    assert pred_severe == 1, f"Expected Severe (1) for Profile 2, got {pred_severe}"
    print("IPF regression check: PASS")

if __name__ == "__main__":
    test_ipf_regression()
