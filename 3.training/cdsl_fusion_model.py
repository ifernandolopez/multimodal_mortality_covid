import os
import pathlib
import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score

# Paths
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
CXR_PATH = SCRIPT_DIR / "../1.cxr_wrangling/cdsl_cxr_features_model.pkl"
EHR_PATH = SCRIPT_DIR / "../2.ehr_data_wrangling/cdsl_structured_ehr.pkl"
FUSION_MODEL_PATH = SCRIPT_DIR / "cdsl_fusion_model.pt"

# Load structured (EHR) features
if not EHR_PATH.exists():
    raise FileNotFoundError(f"EHR features not found: {EHR_PATH}")
ehr_df = pd.read_pickle(EHR_PATH)

# Load CXR features (embeddings)
if not CXR_PATH.exists():
    raise FileNotFoundError(f"CXR features not found: {CXR_PATH}")
cxr_df = joblib.load(CXR_PATH)

# Ensure consistent structure
ehr_df = ehr_df[['patient_id', 'target'] + [col for col in ehr_df.columns if col not in ['patient_id', 'target']]]
cxr_df = cxr_df[['patient_id', 'target'] + [col for col in cxr_df.columns if col not in ['patient_id', 'target']]]

# Merge both modalities
merged = pd.merge(cxr_df, ehr_df, on=["patient_id", "target"], how="inner")
print(f"Fusion dataset shape: {merged.shape}")

# Prepare data
y = merged['target'].values
X = merged.drop(columns=['patient_id', 'target'])
X = X.select_dtypes(include=[np.number]).fillna(0).values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Convert to tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# Define simple FNN
class FusionNet(nn.Module):
    def __init__(self, input_dim):
        super(FusionNet, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.classifier(x)

model = FusionNet(X_train.shape[1])
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training loop
model.train()
for epoch in range(30):
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch + 1}/30, Loss: {loss.item():.4f}")

# Evaluation
model.eval()
with torch.no_grad():
    y_pred_prob = model(X_test_tensor).squeeze().numpy()
    y_pred = (y_pred_prob >= 0.5).astype(int)

print(classification_report(y_test, y_pred, zero_division=0))
print("AUC:", roc_auc_score(y_test, y_pred_prob))

# Save model
torch.save(model.state_dict(), FUSION_MODEL_PATH)
print(f"Fusion model saved to {FUSION_MODEL_PATH}")
