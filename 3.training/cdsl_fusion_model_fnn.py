import os
import pathlib
import joblib
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
import numpy as np

# If you encounter a segmentation fault or GPU memory issues, execute this script with CUDA disabled:
# CUDA_VISIBLE_DEVICES="" python cdsl_fusion_model.py

# Paths
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
EHR_PATH = SCRIPT_DIR / "../1.ehr_data_wrangling/cdsl_structured_ehr.pkl"
CXR_BASE_PATH = SCRIPT_DIR / "../2.cxr_wrangling/cdsl_cxr_features_model.pkl"
CXR_FINETUNE_PATH_PREFIX = SCRIPT_DIR / "../2.cxr_wrangling/cdsl_cxr_finetune_model_"

ehr_data = joblib.load(EHR_PATH)

class FusionFNN(nn.Module):
    def __init__(self, input_dim):
        super(FusionFNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

def evaluate_model(cxr_path, model_name=""):
    print(f"\n=== {model_name} ===")
    cxr_data = joblib.load(cxr_path)

    merged = ehr_data.merge(cxr_data, on="patient_id", suffixes=('_ehr', '_cxr'))

    # Get target column
    target_col = "target_ehr" if "target_ehr" in merged.columns else "target"
    y = merged[target_col].values.astype(np.float32)

    # Drop all target-like columns and patient_id
    drop_cols = [col for col in merged.columns if isinstance(col, str) and "target" in col.lower()] + ["patient_id"]
    X = merged.drop(columns=drop_cols).values.astype(np.float32)


    print(f"Fusion dataset shape: {X.shape}")

    # Split before standardization to avoid data leakage
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Standardize based on training set
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    BATCH_SIZE = 64
    EPOCHS = 30
    LR = 1e-3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FusionFNN(input_dim=X.shape[1]).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model.train()
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device).unsqueeze(1)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss/len(train_loader):.4f}")

    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test).to(device)
        y_pred_prob = model(X_test_tensor).cpu().numpy().flatten()
        y_pred_label = (y_pred_prob >= 0.5).astype(int)

    print(classification_report(y_test, y_pred_label))
    print("AUC:", roc_auc_score(y_test, y_pred_prob))


# Evaluate base model
evaluate_model(CXR_BASE_PATH, model_name="Base")

# Evaluate fine-tuned models
for suffix in ["param355", "layers5", "layers10", "layers50", "block4"]:
    path = pathlib.Path(f"{CXR_FINETUNE_PATH_PREFIX}{suffix}.pkl")
    evaluate_model(path, model_name=f"Fine-tuned {suffix}")
