#!/bin/bash

# Exit on error
set -e

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate net

# Run zero-shot learning scripts
python -m src.benchmark.zero_shot

# Deactivate conda environment
conda deactivate