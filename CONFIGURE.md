# Configuration Guide

This guide provides instructions on how to set up your environment to reproduce the experiments presented in the paper "Developing and Validating Multi-Modal Models for Mortality Prediction in COVID-19 Patients: a Multi-center Retrospective Study."

The following Python libraries and their exact versions are necessary to ensure the stable and error-free execution of the code within this repository, reflecting the environment in which the original experiments were conducted.

## 1. Install a Specific Python Version

pyenv install 3.9.19

## 2. Set Local Python Version

Navigate to the root directory of this repository and set the local Python version. 

pyenv local 3.9.19

## 3. Create and activate a Virtual Environment

Create a virtual environment named .venv:

python -m venv .venv

Activate the virtual environment:

source .venv/bin/activate

## 4. Install Required Libraries


Now, install all the necessary Python libraries and their specific versions. These versions have been tested to be compatible with TensorFlow 2.5.0 and the experiments.

```bash
pip install setuptools # Do not upgrade pip, only install setuptools

pip install \
    tensorflow==2.5.0 \
    h5py==3.1.0 \
    numpy==1.19.5 \
    pandas==1.3.5 \
    matplotlib==3.4.3 \
    scipy==1.7.3 \
    scikit-learn==0.24.2 \
    numba==0.54.0 \
    statsmodels==0.12.2 \
    typing-extensions==3.7.4 \
    gcsfs==2021.4.0 \
    shap==0.39.0 \
    seaborn==0.11.2 \
    tableone==0.8.0 \
    fsspec==2021.04.0 \
    ipykernel==6.0.0 \
    google-cloud-storage==1.36.2 \
    google-api-core==1.26.0 \
    protobuf==3.19.0 \
    requests==2.25.1 \
    pydicom==2.2.2 \
    dill==0.3.4 \
    torch==1.8.1+cpu \
    torchvision==0.9.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
```

## Note on PyTorch: 

The torch and torchvision packages are installed with +cpu to ensure CPU-only versions are used. If you have a compatible GPU and wish to leverage it, please refer to the official PyTorch installation guide for GPU-enabled versions that match your CUDA setup. For example, for CUDA 12.2:

```bash
pip install \
    # ... (rest of the libraries remain the same) \
    torch==1.8.1+cu122 \
    torchvision==0.9.1+cu122 -f https://download.pytorch.org/whl/cu102/torch_stable.html
```
## 5. Configure CDSL dataset path

Before running this script, ensure the 'CDSL_DATA_PATH' environment variable  is set to the absolute path of your downloaded CDSL dataset from PhysioNet.

Example: export CDSL_DATA_PATH=$HOME/physionet.org

The source code will default to your exported path:

DATA_PATH = os.getenv("CDSL_DATA_PATH", "/autor/storage/datasets/physionet.org/files/covid-data-shared-learning/1.0.0/")