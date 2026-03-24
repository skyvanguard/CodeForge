from agent.base_agent import AgentSystem
from agent.llm_withtools import chat_with_agent


class MetaAgent(AgentSystem):
    def forward(self, repo_path, eval_path, iterations_left=None):
        iterations_info = (
            f"\nYou have {iterations_left} remaining iterations to improve the agent."
            if iterations_left is not None
            else ""
        )

        instruction = f"""You are CodeForge's meta-agent: a self-improving system that enhances a Python coding agent.

## Your Goal
Analyze previous evaluation results and modify the task agent to achieve higher test pass rates on Python coding problems.

## Repository
The codebase is at `{repo_path}`. The key file to modify is `task_agent.py`, specifically the `_solve_coding()` method and its prompt.

## Evaluation Results
Previous evaluation results are at `{eval_path}`. Look for:
- `results.json`: per-problem pass/fail results with solution code
- `report.json`: aggregate scores including test_pass_rate and breakdown by difficulty

## Strategy
1. First, read the evaluation results to understand which problems failed and why
2. Read the current task_agent.py to understand the current approach
3. Improve the prompt engineering in _solve_coding() to handle failure patterns:
   - If solutions have syntax errors: add more explicit formatting instructions
   - If solutions are logically wrong: add step-by-step reasoning prompts
   - If solutions miss edge cases: add instructions to consider edge cases
   - If hard problems fail: add chain-of-thought or decomposition prompts
4. You may also modify agent/llm_withtools.py or other agent files if needed
5. Do NOT modify files in the domains/ directory
{iterations_info}

Make targeted, incremental improvements. Test your changes mentally before committing them."""

        new_msg_history = chat_with_agent(
            instruction,
            model=self.model,
            msg_history=[],
            logging=self.log,
            tools_available="all",
        )
