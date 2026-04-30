#!/bin/bash

# Configuration
ENV_NAME="gnn_explainer"
PYTHON_VERSION="3.10"

echo "Creating conda environment: $ENV_NAME"

# 1. Create the environment
conda create -n $ENV_NAME python=$PYTHON_VERSION -y
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME

echo "Installing PyTorch and CUDA toolkit..."
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# 3. Install PyG Dependencies
echo "Installing PyTorch Geometric extensions..."
TORCH_VER=$(python -c "import torch; print(torch.__version__.split('+')[0])")
CUDA_VER=$(python -c "import torch; print('cu' + torch.version.cuda.replace('.', ''))")

pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f "https://data.pyg.org/whl/torch-${TORCH_VER}+${CUDA_VER}.html"

# 4. Install the rest of the requirements
echo "Installing general dependencies..."
pip install torch-geometric lightning networkx scipy pandas scikit-learn matplotlib tqdm click notebook ipykernel

# 5. Install the local project
if [ -f "setup.py" ] || [ -f "pyproject.toml" ]; then
    echo "Installing local project in editable mode..."
    pip install -e .
else
    echo "No setup.py or pyproject.toml found. Skipping 'pip install .'"
fi

echo "Setup complete! Activate with: conda activate $ENV_NAME"#!/bin/bash
