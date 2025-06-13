# Replica el enfoque del paper original,CDLS tiene ficheros JPG, mientras que el estudio usa archivos DICOM como entrada,
# usando su estructura de carpetas y nombres basada en los campos del archivo CDSL-1.0.0-dicom-metadata.csv

import os
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

# Configuración de rutas
DATA_PATH = "/autor/storage/datasets/physionet.org/files/covid-data-shared-learning/1.0.0/"
IMAGE_DIR = os.path.join(DATA_PATH, "IMAGES")
METADATA_CSV = os.path.join(DATA_PATH, "CDSL-1.0.0-dicom-metadata.csv")
PATIENTS_CSV = os.path.join(DATA_PATH, "patient_01.csv")

# Cargar metadatos y pacientes
meta = pd.read_csv(METADATA_CSV, encoding='latin1')
patients = pd.read_csv(PATIENTS_CSV, encoding='latin1')

# Usar solo proyecciones tórax PA o AP
meta = meta[meta['ViewPosition'].isin(['PA', 'AP'])]
print("Total en metadatos (PA/AP):", len(meta))

# Preprocesamiento de imagen
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Cargar modelo DenseNet121 preentrenado
model = densenet121(pretrained=True)
model.classifier = torch.nn.Identity()  # eliminar capa final
model.eval()

# Extraer embeddings
def extract_embedding(path):
    try:
        img = Image.open(path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0)  # shape (1, 3, 224, 224)
        with torch.no_grad():
            embedding = model(img_tensor).squeeze().numpy()
        return embedding
    except Exception as e:
        return None

# Generar embeddings
embeddings = []
ids = []

errores = 0
for i, row in tqdm(meta.iterrows(), total=len(meta)):
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
        errores += 1
print("Embeddings generados (imágenes válidas):", len(embeddings))
print("Errores al abrir imágenes:", errores)

X = pd.DataFrame(embeddings)
X['patient_id'] = ids

# Unir con datos de pacientes
X = X.merge(patients[['patient_id', 'destin_discharge']], on='patient_id', how='left')

# Variable objetivo
X = X.dropna(subset=['destin_discharge'])
y = X['destin_discharge'].apply(lambda x: 1 if str(x).strip().lower() in ['death', 'deceased', 'fallecido'] else 0)

X = X.drop(columns=['patient_id', 'destin_discharge'])

# Mostrar distribución de clases
print("Distribución de la variable objetivo (0 = alta, 1 = fallecido):")
print(y.value_counts())

# Entrenamiento y evaluación
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred, zero_division=0))
print("AUC:", roc_auc_score(y_test, y_prob))

# Guardar el modelo en un archivo .pkl
joblib.dump(clf, "1.cxr_wrangling/cdsl_cxr_features_model.pkl")
print("Modelo guardado en cdsl_cxr_features_model.pkl")
