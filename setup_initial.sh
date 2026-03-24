#!/bin/bash
# ============================================================================
# CodeForge - Setup Script
# ============================================================================
# Verifies prerequisites, pulls Ollama model, builds Docker images,
# and runs initial evaluation.
# ============================================================================

set -e

echo "============================================"
echo "  CodeForge Setup"
echo "============================================"

# 1. Check Ollama
echo ""
echo "[1/5] Checking Ollama..."
if ! command -v ollama &> /dev/null; then
    echo "ERROR: Ollama is not installed. Install from https://ollama.ai"
    exit 1
fi

# Check if Ollama is running
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "ERROR: Ollama is not running. Start it with: ollama serve"
    exit 1
fi
echo "  OK: Ollama is running"

# 2. Check Docker
echo ""
echo "[2/5] Checking Docker..."
if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker is not installed."
    exit 1
fi

if ! docker info > /dev/null 2>&1; then
    echo "ERROR: Docker daemon is not running."
    exit 1
fi
echo "  OK: Docker is running"

# 3. Pull Ollama model
echo ""
echo "[3/5] Pulling Ollama model (qwen2.5-coder:7b)..."
ollama pull qwen2.5-coder:7b
echo "  OK: Model pulled"

# 4. Build Docker images
echo ""
echo "[4/5] Building Docker images..."
docker build -t codeforge .
echo "  OK: codeforge image built"

docker build -t codeforge-runner -f Dockerfile.runner .
echo "  OK: codeforge-runner image built"

# 5. Quick test
echo ""
echo "[5/5] Running quick test..."
python agent/llm.py
echo "  OK: LLM connection works"

echo ""
echo "============================================"
echo "  CodeForge is ready!"
echo ""
echo "  Quick eval:  python -m domains.coding.harness --num_samples 3"
echo "  Full loop:   python generate_loop.py --max_generation 2 --eval_samples 5"
echo "============================================"
