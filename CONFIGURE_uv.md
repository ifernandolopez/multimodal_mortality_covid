# Configuration Guide with uv

This guide describes how to configure the environment using uv, a fast Python package manager that can handle both packages and Python versions. 

uv does not install native system libraries. It relies on the system to provide any required native components (e.g. liblzma, libffi, libopenblas). Therefore, uv should only be used if you have permission to install the necessary system libraries beforehand, or if they are already available on your system.

First of all, clone this repository:

```bash
git clone https://github.com/ifernandolopez/multimodal_mortality_covid
```

## 1. Install python 3.9.19 (first time only)

From the root of the repository

```bash
cd ~/multimodal_mortality_covid
uv init --python 3.9.19
```

Create a .python-version file in the project directory. The pinned version will be used automatically for all python and uv commands within this directory and subdirectories.

## 2.  Create the virtual environment (once)

The following command creates a local virtual environment using the pinned python version.

```bash
uv venv .venv
```

## 3. Activate the virtual environment (every session)

Activate the virtual environment:

```bash
cd ~/multimodal_mortality_covid
source .venv/bin/activate
```

## 5. Install required libraries

Install base tools:

```bash
uv pip install setuptools==58.1.0 wheel==0.45.1 # Do not upgrade pip, only install these packages
```

Install the following versions, which are tested for compatibility with TensorFlow 2.5.0 and the experiments:

```bash
uv pip install \
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
    ipykernel==6.4.0 \
    google-cloud-storage==1.36.2 \
    google-api-core==1.26.0 \
    protobuf==3.19.0 \
    requests==2.25.1 \
    pydicom==2.2.2 \
    dill==0.3.4
```

### Install pytorch

For CPU-only environments, the torch and torchvision packages are installed with +cpu to ensure CPU-only versions are used. 
```bash
uv pip install \
    torch==1.8.1+cpu \
    torchvision==0.9.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
```

If you have a compatible GPU and wish to leverage it, please refer to the official PyTorch installation guide for GPU-enabled versions that match your CUDA setup. For example, for CUDA 11.1:

```bash
uv pip install \
    torch==1.8.1+cu111 \
    torchvision==0.9.1+cu111 -f https://download.pytorch.org/whl/cu111/torch_stable.html
```
## 6. Configure CDSL dataset path

Set the CDSL_DATA_PATH environment variable to the path where you downloaded the dataset from PhysioNet:

```bash
export CDSL_DATA_PATH=$HOME/physionet.org/files/covid-data-shared-learning/1.0.0/
```

The code will then use this path instead of the default:

```python
DATA_PATH = os.getenv("CDSL_DATA_PATH", "/autor/storage/datasets/physionet.org/files/covid-data-shared-learning/1.0.0/")
```

## 7. Run examples

Before running any script, make sure the virtual environment .venv is activated:

```bash
source .venv/bin/activate
```

And optionally conda:

```bash
conda activate python_3_9_19
```

Example for running the structured EHR pipeline:

```bash
cd ~/multimodal_mortality_covid/2.ehr_data_wrangling
python cdsl_structured.py
```
