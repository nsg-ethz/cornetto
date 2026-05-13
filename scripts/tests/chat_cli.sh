#!/bin/bash

#------------------------------------------------------------#
# Simple Bash script to chat with available models with CLI.
# Assumes Ollama is already set up if using Ollama models.
#------------------------------------------------------------#

# Exit on error
set -e

# Set absolute project root directory
export PROJECT_ROOT="$HOME/llm-network-reasoning"
#------------------------------------------------------------#
# Set absolute path to target app
export PATH_TARGET_APP="$PROJECT_ROOT/src/tests/chat_cli.py"
#------------------------------------------------------------#
# CUDA allocation for local models
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Add the project to PYTHONPATH
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Load modules if on a cluster. Otherwise, skip this step.
{
    module purge &&
    module load eth_proxy stack/2024-06 gcc/12.2.0 cuda/11.8
} || {
    echo "Warning! Failed to load required modules"
    echo "Necessary resources should be available on local machine"
}

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate net

# Check CUDA availability
CUDA_VAR=$(python -c 'import torch; print(torch.cuda.is_available())')
echo "CUDA available: $CUDA_VAR"
if [ "$CUDA_VAR" == "True" ]; then
    echo "CUDA version: $(python -c 'import torch; print(torch.version.cuda)')"
    echo "GPU count: $(python -c 'import torch; print(torch.cuda.device_count())')"
    nvidia-smi
fi

# Get model type from config
model_info=$(python -c "
import yaml
with open('$PROJECT_ROOT/configs/zero_shot.yaml', 'r') as f:
    config = yaml.safe_load(f)
print(config.get('model', {}).get('type', 'hf'))
print(config.get('model', {}).get('model_name', ''))
print(config.get('model', {}).get('deployment_mode', 'local'))
")
model_provider=$(echo "$model_info" | head -1)
model_name=$(echo "$model_info" | sed -n '2p')
deployment_mode=$(echo "$model_info" | tail -1)

# Print header
echo "===================================================="
echo "              Starting Chat Application             "
echo "===================================================="
echo "Model Type: $model_provider"  
echo "Model Name: $model_name"
if [ "$model_provider" == "ollama" ]; then
    echo "Deployment mode: $deployment_mode"
fi
echo "===================================================="
echo ""

# Debug info
echo "Started at: $(date)"
echo "Working directory: $(pwd)"
echo "PROJECT_ROOT: $PROJECT_ROOT"
echo "PYTHONPATH: $PYTHONPATH"

# If using Ollama, check if server is running
if [ "$model_provider" == "ollama" ]; then
    # Check if Ollama server is available
    if ! curl -s http://localhost:11434/api/version &> /dev/null; then
        echo "Error: Ollama server is not running!"
        exit 1
    fi
    
    # Check if the model is available
    echo "Checking if model $model_name is available in Ollama..."
    if ! curl -s http://localhost:11434/api/list | grep -q "\"$model_name\""; then
        echo "Error: Model $model_name is not available in Ollama."
        exit 1
    fi
    
    echo "Ollama server is running and model $model_name is available."
fi

# Run the chat application
echo "Starting chat application..."
python "$PATH_TARGET_APP" "$@"

# Completion
echo ""
echo "Closing command-line dialogue"
echo "Finished at: $(date)"

# Deactivate environment
conda deactivate