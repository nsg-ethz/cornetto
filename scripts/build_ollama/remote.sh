#!/bin/bash

#----------------------------------------------------------#
# Script to REMOTELY set up Ollama with Singularity and pull models.
# Run this before chat_cli.sh if using remote Ollama setup.
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

echo "===================================================="
echo "    Setting up Ollama remotely for $model_name      "
echo "===================================================="

# Create an Ollama directory in SCRATCH
OLLAMA_DIR="$SCRATCH/.ollama"
LOG_FILE="$SCRATCH/ollama.log"
TEMP_SCRIPT="$SCRATCH/run_ollama_temp.sh"

# Set up Ollama directory
mkdir -p "$OLLAMA_DIR"

# Temporary Ollama serve script
cat > "$TEMP_SCRIPT" << 'EOF'
#!/bin/bash
export HOME="$SCRATCH"                  # Force HOME to be in SCRATCH
export OLLAMA_LOG_LEVEL=error           # Set higher log level
export OLLAMA_NUM_GPU_LAYERS=50         # Use GPU layers
export OLLAMA_CUDA_MPOOL=1              # Enable CUDA memory pool
cd "$HOME"                              # Change to that directory
mkdir -p .ollama                        # Create .ollama directory
ollama serve                            # Run Ollama server
EOF
    
chmod +x "$TEMP_SCRIPT"

# Verify the Singularity image exists
SINGULARITY_IMAGE="$SCRATCH/ollama-networking-reasoning_latest.sif"
if [ ! -f "$SINGULARITY_IMAGE" ]; then
    echo "Error: Singularity image not found at $SINGULARITY_IMAGE"
    echo "Please build or obtain the Singularity image first."
    exit 1
fi

# Check if Ollama is already running
if curl -s http://localhost:11434/api/version &> /dev/null; then
    echo "Ollama server is already running."
    OLLAMA_RUNNING=true
else
    echo "Starting Ollama server with Singularity..."
    # Start Ollama server with Singularity in the background
    singularity exec --nv --bind "$SCRATCH":"$SCRATCH" "$SINGULARITY_IMAGE" "$TEMP_SCRIPT" > "$LOG_FILE" 2>&1 &
    OLLAMA_PID=$!
    
    echo "Waiting for Ollama server to start..."
    # More robust server check
    for i in {1..30}; do
        if curl -s http://localhost:11434/api/version &> /dev/null; then
            echo "Ollama server started successfully."
            OLLAMA_RUNNING=true
            break
        fi
        if [ $i -eq 30 ]; then
            echo "Ollama server failed to start. Check logs in $LOG_FILE"
            kill $OLLAMA_PID 2>/dev/null || true
            exit 1
        fi
        sleep 1
    done
fi

# Pull the model if needed
echo "Ensuring model $model_name is available..."
if ! singularity exec --nv --bind "$SCRATCH":"$SCRATCH" "$SINGULARITY_IMAGE" ollama list | grep -q "$model_name"; then
    echo "Pulling model $model_name..."
    singularity exec --nv --bind "$SCRATCH":"$SCRATCH" "$SINGULARITY_IMAGE" ollama pull "$model_name" >> "$LOG_FILE" 2>&1
else
    echo "Model $model_name already present."
fi

echo "===================================================="
echo "Ollama setup complete! Server is running and model $model_name is available."
echo "You can now run chat_cli.sh to start chatting."
if [ "$OLLAMA_RUNNING" != "true" ]; then
    echo "To stop Ollama server: kill $OLLAMA_PID"
else
    echo "To stop Ollama server: find and kill the ollama process (ps aux | grep ollama)"
fi
echo "===================================================="