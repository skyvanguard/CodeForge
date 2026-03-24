import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import argparse
import importlib
import importlib.util
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from domains.coding.utils import QUESTION_ID, MODEL, format_input_dict


def load_task_agent(agent_path: str):
    if agent_path.endswith(".py") or os.path.exists(agent_path):
        abs_path = os.path.abspath(agent_path)
        spec = importlib.util.spec_from_file_location("agent_module", abs_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load spec from file: {abs_path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if not hasattr(mod, "TaskAgent"):
            raise AttributeError(f"No TaskAgent found in file: {abs_path}")
        return mod.TaskAgent

    mod = importlib.import_module(agent_path)
    if not hasattr(mod, "TaskAgent"):
        raise AttributeError(f"No TaskAgent found in module: {agent_path}")
    return mod.TaskAgent


def run_agent(TaskAgent, model, problem, evals_folder):
    question_id = problem[QUESTION_ID]
    chat_history_path = os.path.join(evals_folder, f"chat_history_{question_id}.md")
    agent = TaskAgent(model=model, chat_history_file=chat_history_path)
    inputs = format_input_dict(problem)
    prediction, _ = agent.forward(inputs)
    return prediction


def harness(
    agent_path="./task_agent.py",
    output_dir="./outputs",
    run_id=None,
    domain="coding",
    num_samples=-1,
    num_workers=3,
    resume_from=None,
    subset="",
):
    import json

    # Load dataset
    dataset_path = os.path.join(os.path.dirname(__file__), "coding", "dataset.json")
    with open(dataset_path, "r") as f:
        dataset = json.load(f)

    if num_samples > 0:
        dataset = dataset[:num_samples]

    TaskAgent = load_task_agent(agent_path)

    if resume_from:
        output_folder = os.path.abspath(resume_from)
    else:
        run_id = (
            datetime.now().strftime("%Y%m%d_%H%M%S_%f") if run_id is None else run_id
        )
        output_folder = os.path.join(os.getcwd(), output_dir, run_id)

    evals_folder = os.path.join(output_folder, "agent_evals")
    os.makedirs(evals_folder, exist_ok=True)

    results = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for problem in dataset:
            futures.append(
                (problem, executor.submit(run_agent, TaskAgent, MODEL, problem, evals_folder))
            )

        for problem, future in futures:
            prediction = future.result()
            results.append({
                "problem_id": problem["problem_id"],
                "prediction": str(prediction) if prediction else "None",
                "difficulty": problem["difficulty"],
            })

    # Save predictions
    results_path = os.path.join(output_folder, "predictions.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Predictions saved to {results_path}")
    return output_folder


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CodeForge Harness")
    parser.add_argument("--agent_path", type=str, default="./task_agent.py")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--domain", type=str, default="coding", choices=["coding"])
    parser.add_argument("--num_samples", type=int, default=-1)
    parser.add_argument("--num_workers", type=int, default=3)
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--subset", type=str, default="")
    args = parser.parse_args()

    harness(
        agent_path=args.agent_path,
        output_dir=args.output_dir,
        run_id=args.run_id,
        domain=args.domain,
        num_samples=args.num_samples,
        num_workers=args.num_workers,
        resume_from=args.resume_from,
        subset=args.subset,
    )
