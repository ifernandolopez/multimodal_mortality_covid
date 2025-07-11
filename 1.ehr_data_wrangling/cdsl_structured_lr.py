import os
import pathlib
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

# Base path. If the environment variable 'CDSL_DATA_PATH' is not set, it defaults to the specified path
DATA_PATH = os.getenv("CDSL_DATA_PATH", "/autor/storage/datasets/physionet.org/files/covid-data-shared-learning/1.0.0/")
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
EHR_PATH = SCRIPT_DIR / "cdsl_structured_ehr.pkl"
MODEL_PATH = SCRIPT_DIR / "cdsl_structured_model.pkl"

# Load main structured data with safe encoding and no warnings
patients = pd.read_csv(DATA_PATH + "patient_01.csv", encoding='latin1', low_memory=False)
vitals = pd.read_csv(DATA_PATH + "vital_signs_04.csv", encoding='latin1', low_memory=False)
labs = pd.read_csv(DATA_PATH + "lab_06.csv", encoding='latin1', low_memory=False)
meds = pd.read_csv(DATA_PATH + "medication_05.csv", encoding='latin1', low_memory=False)
dx = pd.read_csv(DATA_PATH + "diagnosis_hosp_03.csv", encoding='latin1', low_memory=False)

# Aggregate vital signs and labs by patient
vitals_agg = vitals.groupby('patient_id').mean(numeric_only=True).reset_index()
labs_agg = labs.groupby('patient_id').mean(numeric_only=True).reset_index()

# Merge with patient data
X = patients.merge(vitals_agg, on='patient_id', how='left')
X = X.merge(labs_agg, on='patient_id', how='left')

# Target variable: mortality based on discharge destination
if 'destin_discharge' in patients.columns:
    y = patients['destin_discharge'].apply(
        lambda x: 1 if str(x).strip().lower() in ['death', 'deceased', 'fallecido'] else 0 # Kept 'fallecido' for robustness, can be removed if strict English data expected
    )
else:
    raise ValueError("Column 'destin_discharge' not found to derive the outcome.")

# Prepare dataset for training
X = X.drop(columns=['patient_id', 'hospital_outcome'], errors='ignore')
X = X.select_dtypes(include=[np.number]).fillna(0)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Logistic regression model as baseline
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))
print("AUC:", roc_auc_score(y_test, y_prob))

# Save preprocessed structured data (with patient_id and binary target) for fusion step
structured_df = patients[['patient_id', 'destin_discharge']].copy()
structured_df['target'] = structured_df['destin_discharge'].apply(
    lambda x: 1 if str(x).strip().lower() in ['death', 'deceased', 'fallecido'] else 0
)

# Prepare numerical features (exclude patient_id and target from X; reattach patient_id)
features = X.copy()
features['patient_id'] = patients['patient_id']

# Merge target with features to form the final DataFrame for fusion
ehr_data = pd.merge(structured_df[['patient_id', 'target']], features, on='patient_id')

# Save as pickle for later use in the fusion model
ehr_data.to_pickle(EHR_PATH)
print(f"Structured features saved to {EHR_PATH}")


# Save the structured model
joblib.dump(model, MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")