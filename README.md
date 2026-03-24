# CodeForge

Self-improving Python coding agent that runs 100% local with Ollama.

Based on the self-referential agent loop from [Hyperagents](https://arxiv.org/abs/2603.19461) (Meta Research), adapted to run entirely on local hardware for iterative coding improvement.

## How It Works

```
                    ┌─────────────────────┐
                    │   generate_loop.py   │
                    │  (orchestrator)      │
                    └────────┬────────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
        ┌──────────┐  ┌──────────┐  ┌──────────┐
        │  Select  │  │   Meta   │  │ Evaluate │
        │  Parent  │  │  Agent   │  │  Agent   │
        │ (best    │  │ (reads   │  │ (Docker  │
        │  score)  │  │  results,│  │  sandbox │
        │          │  │  edits   │  │  pytest) │
        └──────────┘  │  task    │  └──────────┘
                      │  agent)  │
                      └──────────┘
```

1. **Task Agent** (`task_agent.py`) solves Python coding problems using a local Ollama model
2. **Evaluator** (`domains/coding/harness.py`) runs solutions in Docker sandboxes with pytest
3. **Meta Agent** (`meta_agent.py`) analyzes results and improves the Task Agent's prompts and code
4. **Loop** (`generate_loop.py`) repeats, keeping an archive of the best-performing generations

Each generation produces a git diff patch. The system tracks lineage, applies cumulative patches, and selects parents using configurable strategies (best score, score-proportional sampling, random, or latest).

## Requirements

- Python 3.12+
- [Ollama](https://ollama.ai) with a coding model (default: `qwen2.5-coder:7b`)
- [Docker](https://docs.docker.com/get-docker/)
- ~8GB VRAM (for 7B model) or ~4GB (for 3B model)

## Quick Start

```bash
# Clone and install
git clone https://github.com/skyvanguard/CodeForge.git
cd CodeForge
pip install -r requirements.txt

# Copy environment config
cp .env.example .env

# Setup: checks Ollama, Docker, pulls model, builds images
bash setup_initial.sh

# Run a quick evaluation (3 problems)
python -m domains.coding.harness --num_samples 3

# Run the self-improvement loop (2 generations, 5 problems each)
python generate_loop.py --max_generation 2 --eval_samples 5
```

## Configuration

Edit `.env` to customize:

```bash
OLLAMA_HOST=http://localhost:11434          # Ollama API endpoint
OLLAMA_MODEL=ollama_chat/qwen2.5-coder:7b  # Model for code generation
```

### Generate Loop Options

```bash
python generate_loop.py \
  --max_generation 10 \        # Number of improvement iterations
  --eval_samples 20 \          # Problems per evaluation (-1 for all 50)
  --eval_workers 3 \           # Parallel evaluation threads
  --parent_selection best \    # Strategy: best, random, latest, score_prop
  --resume_from ./outputs/run_XXX  # Resume a previous run
```

## Project Structure

```
CodeForge/
├── agent/                  # LLM interface and tool-use system
│   ├── llm.py              #   Ollama/litellm integration
│   ├── llm_withtools.py    #   Tool-use loop (parse → execute → respond)
│   ├── base_agent.py       #   Abstract agent base class
│   └── tools/              #   Bash and file editor tools
│       ├── bash.py         #     Persistent bash shell sessions
│       └── edit.py         #     File viewing and editing
├── domains/
│   ├── coding/             # Coding domain
│   │   ├── dataset.json    #   50 Python problems with pytest tests
│   │   ├── harness.py      #   Runs agent + executes tests in sandbox
│   │   ├── report.py       #   Generates score reports
│   │   └── utils.py        #   Domain constants and formatters
│   ├── harness.py          # Generic harness (loads agents dynamically)
│   └── report.py           # Report delegator
├── utils/
│   ├── docker_utils.py     # Container lifecycle management
│   ├── git_utils.py        # Git operations (diff, patch, commit)
│   ├── gl_utils.py         # Generate loop utilities (archive, parent selection)
│   ├── domain_utils.py     # Domain configuration registry
│   ├── constants.py        # Global constants
│   └── common.py           # File I/O and JSON helpers
├── task_agent.py           # Solves coding problems (modified by meta agent)
├── meta_agent.py           # Analyzes results and improves task agent
├── generate_loop.py        # Self-improvement loop orchestrator
├── run_meta_agent.py       # CLI runner for meta agent (used inside Docker)
├── run_task_agent.py       # CLI runner for task agent
├── Dockerfile              # Main container (runs agents with Ollama access)
├── Dockerfile.runner       # Lightweight sandbox for test execution
├── setup_initial.sh        # First-time setup script
└── requirements.txt        # Python dependencies
```

## Dataset

50 Python problems across 3 difficulty levels:

| Difficulty | Count | Examples |
|------------|-------|---------|
| **Easy** | 20 | two_sum, fibonacci, valid_parentheses, reverse_linked_list |
| **Medium** | 21 | binary_search, merge_sort, LRU_cache, coin_change, trie |
| **Hard** | 9 | trapping_rain_water, edit_distance, median_sorted_arrays |

Each problem includes a function signature, description, and pytest test suite. Solutions are executed in isolated Docker containers with network disabled, 256MB memory limit, and 30-second timeout.

## Architecture

### Evaluation Modes

The harness auto-detects its environment:
- **On host**: solutions run in Docker sandbox containers (`codeforge-runner` image) for full isolation
- **Inside Docker**: solutions run via subprocess + pytest directly (the parent container already provides isolation)

### Self-Improvement Loop

```
for each generation:
  1. Select best-performing parent from archive
  2. Create Docker container with project mounted
  3. Apply parent's accumulated patches
  4. Run meta-agent (reads eval results → modifies task_agent.py)
  5. Verify compilation of modified agent
  6. Evaluate modified agent on dataset
  7. Save score, patch, and metadata to archive
  8. Repeat
```

The meta-agent has access to bash and file editing tools. It reads the evaluation results (pass rates, error outputs) and modifies the task agent's prompts, parsing logic, or problem-solving strategy to improve performance.

## Recommended Models

| Model | VRAM | Best For |
|-------|------|----------|
| `qwen2.5-coder:7b` | ~8GB | Default, best quality at 7B |
| `qwen2.5-coder:3b` | ~4GB | Lower VRAM, faster iterations |

## License

MIT License. See [LICENSE](LICENSE) for details.
