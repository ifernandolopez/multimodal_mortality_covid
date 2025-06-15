# Configuration Guide

This guide provides instructions on how to set up your environment to reproduce the experiments presented in the paper "Developing and Validating Multi-Modal Models for Mortality Prediction in COVID-19 Patients: a Multi-center Retrospective Study."

The following Python libraries and their exact versions are necessary to ensure the stable and error-free execution of the code within this repository, reflecting the environment in which the original experiments were conducted.

First of all, clone the repository:

```bash
git clone https://github.com/pyenv/pyenv.git ~/.pyenv
```

## 1. Install python 3.9.19 (first time only)

Choose one of the following options depending on your system permissions:

### Using pyenv

Recommended option if you have permission to install system libraries:

```bash
pyenv install 3.9.19
```

This installs python under $HOME/.pyenv/versions/3.9.19

### Using conda

Use this option if you can't install native libraries system-wide as conda is able to install needed native libraries:

```bash
conda create -n python_3_9_19 python=3.9.19
```

This creates the environment at $HOME/.conda/envs/python_3_9_19

Unlike pyenv, conda can install native dependencies along with python precompiled binaries (called "conda packages") that bundle both the python code and the native libraries they depend on.

So, when you install a package like pandas or numpy with Conda, it automatically includes the necessary native components like liblzma, libffi, or libopenblas, even if those are missing from your system. 

## 2. Activate python 3.9.19

You must activate the correct python version in each session using one of the following options.

### Option 1: Using pyenv (set once)

From the repository directory set the local python version:

```bash
cd ~/multimodal_mortality_covid
pyenv local 3.9.19
```

The option local  activates Python 3.9.19 only within this folder.

### Option 2: Using conda (each session)

With conda you have to activate conda in each session to use python 3.9.19 executing:

```bash
conda activate python_3_9_19
```

And deactivate it at the end of the session with:

```bash
conda deactivate
```

## 3. Create a virtual environment (once)

After activating python 3.9.19, create a virtual environment named .venv in the directory:

```bash
cd ~/multimodal_mortality_covid
python -m venv .venv
```

## 4. Activate the virtual environment (every session)

Activate the virtual environment:

```bash
cd ~/multimodal_mortality_covid
source .venv/bin/activate
```

## 5. Install Required Libraries

Install the following versions, which are tested for compatibility with TensorFlow 2.5.0 and the experiments:

```bash
cd ~/multimodal_mortality_covid

pip install setuptools wheel # Do not upgrade pip, only install setuptools

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
    dill==0.3.4
```

### Install pytorch

For CPU-only environments, the torch and torchvision packages are installed with +cpu to ensure CPU-only versions are used. 
```bash
cd ~/multimodal_mortality_covid

pip install \
    torch==1.8.1+cpu \
    torchvision==0.9.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
```

If you have a compatible GPU and wish to leverage it, please refer to the official PyTorch installation guide for GPU-enabled versions that match your CUDA setup. For example, for CUDA 11.1:

```bash
pip install \
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
