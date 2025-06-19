# Replicates the approach of the original paper. CDLS has JPG files, while the study uses DICOM files as input,
# using its folder structure and names based on the fields of the CDSL-1.0.0-dicom-metadata.csv file.

import os
import pathlib
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image
import joblib
import torch
import torchvision.transforms as transforms
from torchvision.models import densenet121
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

# Base path. If the environment variable 'CDSL_DATA_PATH' is not set, it defaults to the specified path
DATA_PATH = os.getenv("CDSL_DATA_PATH", "/autor/storage/datasets/physionet.org/files/covid-data-shared-learning/1.0.0/")
IMAGE_DIR = os.path.join(DATA_PATH, "IMAGES")
METADATA_CSV = os.path.join(DATA_PATH, "CDSL-1.0.0-dicom-metadata.csv")
PATIENTS_CSV = os.path.join(DATA_PATH, "patient_01.csv")
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
MODEL_PATH = SCRIPT_DIR / "cdsl_cxr_features_model.pkl"

# Load metadata and patient data
meta = pd.read_csv(METADATA_CSV, encoding='latin1')
patients = pd.read_csv(PATIENTS_CSV, encoding='latin1')

# Use only PA or AP chest projections
meta = meta[meta['ViewPosition'].isin(['PA', 'AP'])]
print("Total in metadata (PA/AP):", len(meta))

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load pre-trained DenseNet121 model
model = densenet121(pretrained=True)
model.classifier = torch.nn.Identity()  # remove final layer
model.eval()

# Extract embeddings
def extract_embedding(path):
    try:
        img = Image.open(path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0)  # shape (1, 3, 224, 224)
        with torch.no_grad():
            embedding = model(img_tensor).squeeze().numpy()
        return embedding
    except Exception:
        return None

# Generate embeddings
embeddings = []
ids = []

errors = 0
for _, row in tqdm(meta.iterrows(), total=len(meta)):
    img_path = os.path.join(
        IMAGE_DIR,
        row['patient_group_folder_id'],
        row['patient_folder_id'],
        row['study_id'],
        row['image_id'] + ".jpg")
    emb = extract_embedding(img_path)
    if emb is not None:
        embeddings.append(emb)
        ids.append(row['patient_id'])
    else:
        errors += 1
print("Embeddings generated (valid images):", len(embeddings))
print("Errors opening images:", errors)

# Create DataFrame with embeddings and IDs
embedding_df = pd.DataFrame(embeddings)
embedding_df['patient_id'] = ids

# Merge with patient data
embedding_df = embedding_df.merge(patients[['patient_id', 'destin_discharge']], on='patient_id', how='left')

# Filter and define target variable
embedding_df = embedding_df.dropna(subset=['destin_discharge'])
embedding_df['target'] = embedding_df['destin_discharge'].apply(
    lambda x: 1 if str(x).strip().lower() in ['death', 'deceased', 'fallecido'] else 0
)

# Prepare feature matrix and labels
X = embedding_df.drop(columns=['patient_id', 'destin_discharge', 'target'])
y = embedding_df['target']

# Show class distribution
print("Target variable distribution (0 = discharged, 1 = deceased):")
print(y.value_counts())

# Training and evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred, zero_division=0))
print("AUC:", roc_auc_score(y_test, y_prob))

# Reorder columns for saving
cols = ['patient_id', 'target'] + [col for col in X.columns]
embedding_df = embedding_df[cols]

# Save final DataFrame
joblib.dump(embedding_df, MODEL_PATH)
print(f"Embeddings with labels saved to {MODEL_PATH}")
