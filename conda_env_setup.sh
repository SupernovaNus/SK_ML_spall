#!/bin/bash

# Execute this in the local of OSC to setup the environment (don't have to submit a job)

module load miniconda3

# Create a new Conda environment
conda create --name spall_ml python=3.8 -y

# Find the absolute path to the environment's directory
ENV_DIR=$(conda env list | grep 'spall_ml' | awk '{print $2}')

# Use the environment's pip executable to install packages
${ENV_DIR}/bin/pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
${ENV_DIR}/bin/pip install -r requirements.txt