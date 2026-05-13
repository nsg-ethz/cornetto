#!/bin/bash

#----------------------------------------------------------#
# Script to LOCALLY set up Ollama and pull models.
# Run this before chat_cli.sh if using a local Ollama setup.
#----------------------------------------------------------#

# Exit on error
set -e

# Set project root
if [ -z "$PROJECT_ROOT" ]; then
    export PROJECT_ROOT="$HOME/llm-network-reasoning"
fi

# Get model name as argument or from config
if [ -n "$1" ]; then
    model_name="$1"
else
    # Get from config
    model_info=$(python -c "
    import yaml
    with open('$PROJECT_ROOT/configs/zero_shot.yaml', 'r') as f:
        config = yaml.safe_load(f)
    print(config.get('model', {}).get('model_name', ''))
    ")
    model_name="$model_info"
fi

LOG_FILE="$SCRATCH/ollama_cluster.log"

echo "===================================================="
echo "       Setting up Ollama locally for $model_name    "
echo "===================================================="

# Verify that Ollama is installed and available
if ! command -v ollama &> /dev/null; then
    echo "Error: Ollama is not installed or not in PATH."
    echo "Please install Ollama on the cluster or switch to 'singularity' deployment mode."
    exit 1
fi

# Check if Ollama server is running
if ! curl -s http://localhost:11434/api/version &> /dev/null; then
    echo "Starting Ollama server..."
    # Set environment variables for Ollama
    export OLLAMA_LOG_LEVEL=error           # Set higher log level
    export OLLAMA_NUM_GPU_LAYERS=50         # Use GPU layers
    export OLLAMA_CUDA_MPOOL=1              # Enable CUDA memory pool
    
    # Start Ollama in the background
    ollama serve > "$LOG_FILE" 2>&1 &
    OLLAMA_PID=$!
    
    # Wait for Ollama to start
    echo "Waiting for Ollama server to start..."
    for i in {1..30}; do
        if curl -s http://localhost:11434/api/version &> /dev/null; then
            echo "Ollama server started successfully."
            break
        fi
        if [ $i -eq 30 ]; then
            echo "Ollama server failed to start. Check logs in $LOG_FILE"
            kill $OLLAMA_PID 2>/dev/null || true
            exit 1
        fi
        sleep 1
    done
else
    echo "Ollama server is already running."
fi

# Check if model is available
echo "Checking if model $model_name is available..."
if ! ollama list | grep -q "$model_name"; then
    echo "Model $model_name not found. Pulling model..."
    ollama pull "$model_name"
else
    echo "Model $model_name is already available."
fi

echo "===================================================="
echo "Ollama setup complete! Server is running and model $model_name is available."
echo "You can now run chat_cli.sh to start chatting."
echo "To stop Ollama server: pkill -f 'ollama serve'"
echo "===================================================="