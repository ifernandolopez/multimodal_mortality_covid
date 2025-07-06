import os
import pathlib
import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score

# Base path. If the environment variable 'CDSL_DATA_PATH' is not set, it defaults to the specified path
DATA_PATH = os.getenv("CDSL_DATA_PATH", "/autor/storage/datasets/physionet.org/files/covid-data-shared-learning/1.0.0/")
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
EHR_PATH = SCRIPT_DIR / "cdsl_structured_ehr.pkl"
MODEL_PATH = SCRIPT_DIR / "cdsl_structured_model.pt"

# Load data
patients = pd.read_csv(DATA_PATH + "patient_01.csv", encoding='latin1', low_memory=False)
vitals = pd.read_csv(DATA_PATH + "vital_signs_04.csv", encoding='latin1', low_memory=False)
labs = pd.read_csv(DATA_PATH + "lab_06.csv", encoding='latin1', low_memory=False)

# Aggregate
vitals_agg = vitals.groupby('patient_id').mean(numeric_only=True).reset_index()
labs_agg = labs.groupby('patient_id').mean(numeric_only=True).reset_index()

# Merge
X = patients.merge(vitals_agg, on='patient_id', how='left')
X = X.merge(labs_agg, on='patient_id', how='left')

# Target
if 'destin_discharge' in patients.columns:
    y = patients['destin_discharge'].apply(
        lambda x: 1 if str(x).strip().lower() in ['death', 'deceased', 'fallecido'] else 0
    )
else:
    raise ValueError("Column 'destin_discharge' not found.")

# Prepare data
X = X.drop(columns=['patient_id', 'hospital_outcome'], errors='ignore')
X = X.select_dtypes(include=[np.number]).fillna(0)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Convert to tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

# Model
class SimpleFNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)  # No Sigmoid here
        )

    def forward(self, x):
        return self.net(x)

model = SimpleFNN(X_train.shape[1])
pos_weight = torch.tensor([len(y_train) / y_train.sum() - 1])
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train
for epoch in range(30):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}/30, Loss: {loss.item():.4f}")

# Evaluate
model.eval()
with torch.no_grad():
    logits = model(X_test_tensor)
    probs = torch.sigmoid(logits).numpy().flatten()
    preds = (probs > 0.5).astype(int)

print(classification_report(y_test, preds, zero_division=0))
print("AUC:", roc_auc_score(y_test, probs))

# Save preprocessed structured data for fusion
structured_df = patients[['patient_id', 'destin_discharge']].copy()
structured_df['target'] = structured_df['destin_discharge'].apply(
    lambda x: 1 if str(x).strip().lower() in ['death', 'deceased', 'fallecido'] else 0
)

features = pd.DataFrame(X_scaled, columns=X.columns)
features['patient_id'] = patients['patient_id']

ehr_data = pd.merge(structured_df[['patient_id', 'target']], features, on='patient_id')
ehr_data.to_pickle(EHR_PATH)
print(f"Structured features saved to {EHR_PATH}")

# Save model
torch.save(model.state_dict(), MODEL_PATH)
print(f"FNN model saved to {MODEL_PATH}")
