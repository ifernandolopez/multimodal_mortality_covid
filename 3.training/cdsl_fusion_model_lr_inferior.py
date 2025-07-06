import os
import pathlib
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

# Paths
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
CXR_PATH = SCRIPT_DIR / "../1.cxr_wrangling/cdsl_cxr_features_model.pkl"
EHR_PATH = SCRIPT_DIR / "../2.ehr_data_wrangling/cdsl_structured_ehr.pkl"
FUSION_MODEL_PATH = SCRIPT_DIR / "cdsl_fusion_model.pkl"

# Load structured (EHR) features
if not EHR_PATH.exists():
    raise FileNotFoundError(f"EHR features not found: {EHR_PATH}")
ehr_df = pd.read_pickle(EHR_PATH)

# Load CXR features (embeddings)
if not CXR_PATH.exists():
    raise FileNotFoundError(f"CXR features not found: {CXR_PATH}")
cxr_df = joblib.load(CXR_PATH)

# Ensure both contain the same 'patient_id' and 'target'
ehr_df = ehr_df[['patient_id', 'target'] + [col for col in ehr_df.columns if col not in ['patient_id', 'target']]]
cxr_df = cxr_df[['patient_id', 'target'] + [col for col in cxr_df.columns if col not in ['patient_id', 'target']]]

# Merge both modalities on patient_id and target
merged = pd.merge(cxr_df, ehr_df, on=["patient_id", "target"], how="inner")

print(f"Fusion dataset shape: {merged.shape}")

# Prepare features and labels
y = merged['target']
X = merged.drop(columns=['patient_id', 'target'])
X = X.select_dtypes(include=[np.number]).fillna(0)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Train fusion model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred, zero_division=0))
print("AUC:", roc_auc_score(y_test, y_prob))

# Save fusion model
joblib.dump(model, FUSION_MODEL_PATH)
print(f"Fusion model saved to {FUSION_MODEL_PATH}")
