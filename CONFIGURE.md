# Configuration Guide

This guide provides instructions on how to set up your environment to reproduce the experiments presented in the paper "Developing and Validating Multi-Modal Models for Mortality Prediction in COVID-19 Patients: a Multi-center Retrospective Study."

The following Python libraries and their exact versions are necessary to ensure the stable and error-free execution of the code within this repository, reflecting the environment in which the original experiments were conducted.

You can choose among three setup methods depending on your system permissions and preferences:

- Option A: pyenv + pip (legacy, traditional approach)
- Option B: uv (modern and fast package/environment manager)
- Option C: conda (recommended if you can't install native system libraries)

All options aim to install Python 3.9.19 and the same set of dependencies.

## 1. Clone the repository

```bash
git clone https://github.com/ifernandolopez/multimodal_mortality_covid
```

## 2. Install python 3.9.19 (first time only)

There are three supported methods to install python 3.9.19 depending on your system configuration and permissions:

### Option 1: pyenv (legacy, traditional approach)

Use this if you have permission to install system-level libraries (e.g. libffi, liblzma, etc.):

```bash
pyenv install 3.9.19
```

This installs Python under $HOME/.pyenv/versions/3.9.19. You can then activate it locally by running pyenv local 3.9.19 in the repository directory.

### Option 2: uv (modern, fast package manager)

Use this if you prefer a reproducible, modern uv workflow. Use only if you have permission to install native libraries.

From the root of the repository:

```bash
cd multimodal_mortality_covid
uv init --python 3.9.19
```
Create a .python-version file in the project directory. The pinned version will be used automatically for all python and uv commands within this directory and subdirectories.


### Option 3: Using conda

Use this option if you can't install native libraries system-wide (e.g., on a shared server without root access) because conda is able to install needed native libraries:

```bash
conda create -n python_3_9_19 python=3.9.19
```

This creates the environment at $HOME/.conda/envs/python_3_9_19

Unlike pyenv or uv, conda is able to install native dependencies along with python precompiled binaries (called "conda packages") that bundle both the python code and the native libraries they depend on.

So, when you install a package like pandas or numpy with Conda, it automatically includes the necessary native components like liblzma, libffi, or libopenblas, even if those are missing from your system. 

## 3. Activate python 3.9.19

You must activate the correct python version in each session using one of the following options.

### Option 1: Using pyenv

From the repository directory set the local python version:

```bash
cd multimodal_mortality_covid
pyenv local 3.9.19
```

This activates Python 3.9.19 automatically whenever you're inside this folder. The version is pinned in .python-version.:

```bash
cat ./vstudio/multimodal_mortality_covid/.python-version
3.9.19
```

### Option 2: Using uv

No manual activation is needed. The pinned python version in .python-version is used automatically when inside the project directory.

### Option 3: Using conda

You need to activate the environment manually at the start of each session:

```bash
conda activate python_3_9_19
```

And deactivate it at the end of the session:

```bash
conda deactivate
```

## 4. Create a virtual environment (first time only)

After activating python 3.9.19, create a virtual environment named .venv in the directory. The steps differ slightly depending on the setup tool.

### Option 1 or 3: pyenv or conda

Once python 3.9.19 is active (via pyenv local or conda activate), create a standard virtual environment in the project directory:

```bash
python -m venv .venv
```

### Option 2: uv

If you're using uv, create the virtual environment as follows:

```bash
uv venv .venv
```

## 5. Activate the virtual environment (every session)

Before using the environment, you must activate it at the beginning of every session.

If you are using conda, activate the conda python 3.9.19 environment first:

```bash
conda activate python_3_9_19
```

Then, activate the virtual environment (applies to all options):

```bash
cd multimodal_mortality_covid
source .venv/bin/activate
```

## 6. Install required libraries (first time only)

Use `pip` with pyenv or conda, and `uv pip` with uv.

Install base tools. Do not upgrade pip, only install these packages:

```bash
pip install setuptools==58.1.0 wheel==0.45.1
# or
uv pip install setuptools==58.1.0 wheel==0.45.1
```

Install the following versions, which are tested for compatibility with TensorFlow 2.5.0 and the experiments:

```bash
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
    ipykernel==6.4.0 \
    google-cloud-storage==1.36.2 \
    google-api-core==1.26.0 \
    protobuf==3.19.0 \
    requests==2.25.1 \
    pydicom==2.2.2 \
    dill==0.3.4
```

Replace pip with uv pip if using uv.

### Install pytorch

#### Option A: CUDA version

If you have a compatible GPU and wish to leverage it, please refer to the official PyTorch installation guide for GPU-enabled versions that match your CUDA setup. For example, for CUDA 11.1:

```bash
pip install \
    torch==1.8.1+cu111 \
    torchvision==0.9.1+cu111 -f https://download.pytorch.org/whl/cu111/torch_stable.html
```

#### Option B: CPU version

For CPU-only environments, the torch and torchvision packages are installed with +cpu to ensure CPU-only versions are used. 

```bash
pip install \
    torch==1.8.1+cpu \
    torchvision==0.9.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
```

Use uv pip if you're managing packages with uv.

## 7. Configure CDSL dataset path

Set the CDSL_DATA_PATH environment variable to the path where you downloaded the dataset from PhysioNet:

```bash
export CDSL_DATA_PATH=$HOME/physionet.org/files/covid-data-shared-learning/1.0.0/
```

The code will then use this path instead of the default:

```python
DATA_PATH = os.getenv("CDSL_DATA_PATH", "/autor/storage/datasets/physionet.org/files/covid-data-shared-learning/1.0.0/")
```

## 8. Run examples

Make sure .venv is still active. If not, activate it again with:

```bash
source .venv/bin/activate
```

And optionally conda:

```bash
conda activate python_3_9_19
```

Example for running the structured EHR pipeline:

```bash
cd multimodal_mortality_covid/2.ehr_data_wrangling
python cdsl_structured.py
```
