import os
import pathlib
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
EHR_PATH = SCRIPT_DIR / "../1.ehr_data_wrangling/cdsl_structured_ehr.pkl"
CXR_BASE_PATH = SCRIPT_DIR / "../2.cxr_wrangling/cdsl_cxr_features_model.pkl"
CXR_FINETUNE_PATH_PREFIX = SCRIPT_DIR / "../2.cxr_wrangling/cdsl_cxr_finetune_model_"

# Load structured EHR data
ehr_data = pd.read_pickle(EHR_PATH)

def evaluate_model(cxr_path, model_name=""):
    try:
        cxr_data = joblib.load(cxr_path)
    except FileNotFoundError:
        print(f"{model_name} file not found: {cxr_path}")
        return

    # Merge on patient_id
    fusion = pd.merge(ehr_data, cxr_data, on=["patient_id", "target"], how="inner")
    print(f"{model_name} Fusion dataset shape:", fusion.shape)

    # Prepare features and target
    X = fusion.drop(columns=["patient_id", "target"])
    y = fusion["target"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    print(f"Results for {model_name}")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("AUC:", roc_auc_score(y_test, y_prob))
    print("")

# Evaluate base model
evaluate_model(CXR_BASE_PATH, model_name="Base")

# Evaluate fine-tuned models
for suffix in ["last10", "last50"]:
    path = pathlib.Path(f"{CXR_FINETUNE_PATH_PREFIX}{suffix}.pkl")
    evaluate_model(path, model_name=f"Fine-tuned {suffix}")
