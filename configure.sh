#!/bin/bash

# This script must be run with 'source configure.sh' (not './configure.sh')
# to ensure that the environment activation persists in the current shell session.

# Load Conda into the shell session
source /opt/tljh/user/etc/profile.d/conda.sh

# Activate Conda environment
conda activate python_3_9_19

# Activate local virtual environment
source .venv/bin/activate

