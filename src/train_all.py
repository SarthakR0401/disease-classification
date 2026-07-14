import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import os
import sys

# Reconfigure stdout to prevent encoding crashes with emojis
sys.stdout.reconfigure(encoding='utf-8')

notebooks_dir = r"d:\Esparse Matrix\IPYNB"
notebooks = ["ASTHMA.ipynb", "COPD.ipynb", "IPF.ipynb", "Pneumonia.ipynb"]

print("Starting programmatic notebook training run...\n")

for nb_name in notebooks:
    nb_path = os.path.join(notebooks_dir, nb_name)
    print(f"=== Running {nb_name} ===")
    if not os.path.exists(nb_path):
        print(f"Error: {nb_path} does not exist!")
        continue
    try:
        with open(nb_path, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)
        
        # Execute the notebook
        ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
        # Run it in its own directory so relative paths work
        ep.preprocess(nb, {"metadata": {"path": notebooks_dir}})
        
        # Write the executed notebook back
        with open(nb_path, "w", encoding="utf-8") as f:
            nbformat.write(nb, f)
        print(f"Successfully executed {nb_name} and saved results.\n")
    except Exception as e:
        print(f"Error running {nb_name}: {e}\n")

print("\n=== Model Verification ===")
expected_files = [
    "best_asthma_clinical_model.pkl", "asthma_scaler.pkl", "asthma_feature_columns.pkl",
    "best_copd_clinical_model.pkl", "copd_scaler.pkl", "copd_feature_columns.pkl",
    "best_ipf_clinical_model.pkl", "ipf_scaler.pkl", "ipf_label_encoders.pkl",
    "best_pneumonia_clinical_model.pkl", "pneumonia_scaler.pkl", "pneumonia_imputation_values.pkl",
    "pneumonia_feature_columns.pkl", "pneumonia_clinical_model_bundle.pkl"
]

for f in expected_files:
    f_path = os.path.join(notebooks_dir, f)
    status = "✅ Found" if os.path.exists(f_path) else "❌ Missing"
    print(f"  {f:<40} : {status}")
