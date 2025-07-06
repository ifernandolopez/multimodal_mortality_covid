import os
import pathlib
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models import densenet121
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

# Paths
DATA_PATH = os.getenv("CDSL_DATA_PATH", "/autor/storage/datasets/physionet.org/files/covid-data-shared-learning/1.0.0/")
IMAGE_DIR = os.path.join(DATA_PATH, "IMAGES")
METADATA_CSV = os.path.join(DATA_PATH, "CDSL-1.0.0-dicom-metadata.csv")
PATIENTS_CSV = os.path.join(DATA_PATH, "patient_01.csv")
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
MODEL_PATH_PREFIX = SCRIPT_DIR / f"cdsl_cxr_finetune_model_"

# Parameters
BATCH_SIZE = 16 # Reduced to limit GPU/CPU memory usage
EPOCHS = 10
LR = 1e-4

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load metadata
meta = pd.read_csv(METADATA_CSV, encoding='latin1')
patients = pd.read_csv(PATIENTS_CSV, encoding='latin1')

meta = meta[meta['ViewPosition'].isin(['PA', 'AP'])]

# Merge with patient outcome
targets = patients[['patient_id', 'destin_discharge']].copy()
targets['target'] = targets['destin_discharge'].apply(
    lambda x: 1 if str(x).strip().lower() in ['death', 'deceased', 'fallecido'] else 0
)
meta = meta.merge(targets[['patient_id', 'target']], on='patient_id', how='left')

# Dataset
class CXRDataset(Dataset):
    def __init__(self, df, image_dir, transform):
        self.df = df
        self.image_dir = image_dir
        self.transform = transform
        self.valid_indices = []
        for i in range(len(df)):
            row = df.iloc[i]
            img_path = os.path.join(
                self.image_dir,
                row['patient_group_folder_id'],
                row['patient_folder_id'],
                row['study_id'],
                row['image_id'] + ".jpg"
            )
            if os.path.exists(img_path):
                self.valid_indices.append(i)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        row = self.df.iloc[self.valid_indices[idx]]
        img_path = os.path.join(
            self.image_dir,
            row['patient_group_folder_id'],
            row['patient_folder_id'],
            row['study_id'],
            row['image_id'] + ".jpg"
        )
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        label = torch.tensor(row['target'], dtype=torch.float32)
        return image, label

# Split data
train_df, test_df = train_test_split(meta, test_size=0.2, stratify=meta['target'], random_state=42)
train_dataset = CXRDataset(train_df, IMAGE_DIR, transform)
test_dataset = CXRDataset(test_df, IMAGE_DIR, transform)
# Set num_workers=0 and pin_memory=False to prevent excessive memory use
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)

# Model setup
def finetune_and_save_embeddings(last_unfreeze_layer: int, name_suffix: str):
    model = densenet121(pretrained=True)
    num_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_features, 1),
        nn.Sigmoid()
    )

    # Freeze all layers first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze last X parameters
    trainable_params = list(model.parameters())[-last_unfreeze_layer:]
    for param in trainable_params:
        param.requires_grad = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)

    # Training loop
    model.train()
    for epoch in range(EPOCHS):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device).unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {running_loss/len(train_loader):.4f}")
        torch.cuda.empty_cache() # Free up unused GPU memory after each epoch

    # Embedding extraction controlling memory usage
    model.classifier = nn.Identity()
    model.eval()
    embeddings, ids = [], []

    eval_dataset = CXRDataset(meta, IMAGE_DIR, transform)
    eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)

    with torch.no_grad():
        for images, _ in tqdm(eval_loader, total=len(eval_loader)):
            images = images.to(device)
            embs = model(images).cpu().numpy()
            batch_ids = meta.iloc[eval_dataset.valid_indices[:len(embs)]]['patient_id'].values
            embeddings.extend(embs)
            ids.extend(batch_ids)
            eval_dataset.valid_indices = eval_dataset.valid_indices[len(embs):]

    emb_df = pd.DataFrame(embeddings)
    emb_df['patient_id'] = ids
    emb_df = emb_df.merge(targets[['patient_id', 'target']], on='patient_id', how='left')
    cols = ['patient_id', 'target'] + [col for col in emb_df.columns if col not in ['patient_id', 'target']]
    emb_df = emb_df[cols]
    outfile = pathlib.Path(str(MODEL_PATH_PREFIX) + f"{name_suffix}.pkl")
    joblib.dump(emb_df, outfile)
    print(f"Saved finetuned embeddings to {outfile}")

# Selected finetunning deeps
finetune_and_save_embeddings(last_unfreeze_layer=9, name_suffix="last10")
finetune_and_save_embeddings(last_unfreeze_layer=8, name_suffix="last50")
