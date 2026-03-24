# CodeForge

Self-improving Python coding agent that runs 100% local with Ollama.

Based on the self-referential agent loop from [Hyperagents](https://arxiv.org/abs/2603.19461) (Meta Research), adapted to run entirely on local hardware for iterative coding improvement.

## How it works

1. **Task Agent** solves Python coding problems using an Ollama model
2. **Evaluator** runs solutions in Docker sandboxes with pytest
3. **Meta Agent** analyzes results and improves the Task Agent's prompts/code
4. **Loop** repeats — each generation potentially better than the last

## Requirements

- Python 3.12+
- [Ollama](https://ollama.ai) with a coding model (default: `qwen2.5-coder:7b`)
- Docker

## Quick Start

```bash
# Clone and install
git clone https://github.com/skyvanguard/CodeForge.git
cd CodeForge
pip install -r requirements.txt

# Setup (checks Ollama, Docker, pulls model, builds images)
bash setup_initial.sh

# Run a quick evaluation (3 problems)
python -m domains.coding.harness --num_samples 3

# Run the self-improvement loop (2 generations, 5 problems each)
python generate_loop.py --max_generation 2 --eval_samples 5
```

## Configuration

Copy `.env.example` to `.env` and adjust:

```bash
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=ollama_chat/qwen2.5-coder:7b
```

## File Structure

```
agent/           LLM interface and tool-use system
domains/coding/  Dataset (50 problems), harness, report
utils/           Docker, git, domain config utilities
task_agent.py    Solves coding problems (modified by meta agent)
meta_agent.py    Analyzes results and improves task agent
generate_loop.py Self-improvement loop orchestrator
```

## Dataset

50 Python problems across 3 difficulty levels:
- **Easy** (20): two_sum, fibonacci, valid_parentheses, etc.
- **Medium** (20): binary_search, merge_sort, LRU cache, coin_change, etc.
- **Hard** (10): trapping_rain_water, edit_distance, median_sorted_arrays, etc.

Solutions are executed in isolated Docker containers with network disabled, 256MB memory limit, and 30s timeout.

## Recommended Model

**Qwen2.5-Coder:7b** — best coding model at 7B parameters, fits in 8GB VRAM, good tool-use, 128K context.
