import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

# Ruta base
DATA_PATH = "/autor/storage/datasets/physionet.org/files/covid-data-shared-learning/1.0.0/"

# Cargar datos estructurados principales con codificación segura y sin warnings
patients = pd.read_csv(DATA_PATH + "patient_01.csv", encoding='latin1', low_memory=False)
vitals = pd.read_csv(DATA_PATH + "vital_signs_04.csv", encoding='latin1', low_memory=False)
labs = pd.read_csv(DATA_PATH + "lab_06.csv", encoding='latin1', low_memory=False)
meds = pd.read_csv(DATA_PATH + "medication_05.csv", encoding='latin1', low_memory=False)
dx = pd.read_csv(DATA_PATH + "diagnosis_hosp_03.csv", encoding='latin1', low_memory=False)

# Previsualización
print(patients.head())
print(vitals.head())

# Agregar signos vitales y laboratorios por paciente
vitals_agg = vitals.groupby('patient_id').mean(numeric_only=True).reset_index()
labs_agg = labs.groupby('patient_id').mean(numeric_only=True).reset_index()

# Unir con datos de pacientes
X = patients.merge(vitals_agg, on='patient_id', how='left')
X = X.merge(labs_agg, on='patient_id', how='left')

# Variable objetivo: mortalidad según destino al alta
if 'destin_discharge' in patients.columns:
    y = patients['destin_discharge'].apply(
        lambda x: 1 if str(x).strip().lower() in ['death', 'deceased', 'fallecido'] else 0
    )
else:
    raise ValueError("No se encuentra la columna 'destin_discharge' para derivar el outcome")

# Preparar conjunto para entrenamiento
X = X.drop(columns=['patient_id', 'hospital_outcome'], errors='ignore')
X = X.select_dtypes(include=[np.number]).fillna(0)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Modelo de regresión logística como baseline
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluación
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))
print("AUC:", roc_auc_score(y_test, y_prob))

# Guardar el modelo estructurado
joblib.dump(model, "2.ehr_data_wrangling/cdsl_structured_model.pkl")
print("Modelo guardado en cdsl_model.pkl")
