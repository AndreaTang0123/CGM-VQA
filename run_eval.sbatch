#!/bin/bash
#SBATCH --job-name=cgm_vqa_eval
#SBATCH --output=logs/eval_%j.log
#SBATCH --error=logs/eval_%j.err
#SBATCH --time=08:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

# Create logs directory if it doesn't exist
mkdir -p logs

# Load necessary modules (adjust based on your cluster)
module load python

cd "$(dirname "$0")"

# Start Ollama in background
echo "Starting Ollama service..."
ollama serve &
OLLAMA_PID=$!

# Wait for Ollama to be ready
echo "Waiting for Ollama to start..."
sleep 10
for i in {1..30}; do
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "Ollama is ready!"
        break
    fi
    echo "Attempt $i: Waiting for Ollama..."
    sleep 5
done

# Pull the model if needed
echo "Ensuring llava:7b model is available..."
ollama pull llava:7b

# Run the evaluation script
echo "Starting evaluation..."
python run_eval_llava7b.py

# Cleanup
echo "Shutting down Ollama..."
kill $OLLAMA_PID
wait $OLLAMA_PID 2>/dev/null

echo "Evaluation complete!"
