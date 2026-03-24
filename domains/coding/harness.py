import argparse
import json
import os
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import docker

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from domains.coding.utils import QUESTION_ID, MODEL, format_input_dict


def load_dataset(num_samples=-1):
    dataset_path = os.path.join(os.path.dirname(__file__), "dataset.json")
    with open(dataset_path, "r") as f:
        dataset = json.load(f)
    if num_samples > 0:
        dataset = dataset[:num_samples]
    return dataset


def run_solution_in_docker(client, solution_code, test_code, timeout=30):
    try:
        container = client.containers.run(
            image="codeforge-runner",
            command=["pytest", "-v", "--tb=short", "/sandbox/test_solution.py"],
            volumes={},
            detach=True,
            network_disabled=True,
            mem_limit="256m",
        )

        # Copy solution and test files into container
        import io
        import tarfile

        tar_stream = io.BytesIO()
        with tarfile.open(fileobj=tar_stream, mode="w") as tar:
            # Add solution.py
            sol_data = solution_code.encode("utf-8")
            sol_info = tarfile.TarInfo(name="solution.py")
            sol_info.size = len(sol_data)
            tar.addfile(sol_info, io.BytesIO(sol_data))

            # Add test_solution.py
            test_data = test_code.encode("utf-8")
            test_info = tarfile.TarInfo(name="test_solution.py")
            test_info.size = len(test_data)
            tar.addfile(test_info, io.BytesIO(test_data))

        tar_stream.seek(0)
        container.put_archive("/sandbox", tar_stream)

        # Execute tests
        exit_code, output = container.exec_run(
            ["pytest", "-v", "--tb=short", "/sandbox/test_solution.py"],
            workdir="/sandbox",
        )

        output_text = output.decode("utf-8") if isinstance(output, bytes) else str(output)

        # Parse results
        passed = output_text.count(" PASSED")
        failed = output_text.count(" FAILED")
        errors = output_text.count(" ERROR")
        total = passed + failed + errors

        container.stop(timeout=5)
        container.remove(force=True)

        return {
            "passed": passed,
            "failed": failed,
            "errors": errors,
            "total": total,
            "output": output_text[:2000],
            "exit_code": exit_code,
        }

    except Exception as e:
        return {
            "passed": 0,
            "failed": 0,
            "errors": 1,
            "total": 1,
            "output": str(e)[:2000],
            "exit_code": -1,
        }


def evaluate_problem(problem, agent_path, docker_client):
    from domains.coding.utils import format_input_dict
    from domains.harness import load_task_agent

    TaskAgent = load_task_agent(agent_path)

    inputs = format_input_dict(problem)
    chat_history_path = os.path.join(
        tempfile.gettempdir(), f"chat_history_{problem['problem_id']}.md"
    )
    agent = TaskAgent(model=MODEL, chat_history_file=chat_history_path)
    prediction, _ = agent.forward(inputs)

    solution_code = str(prediction) if prediction else ""

    # Run tests in Docker sandbox
    result = run_solution_in_docker(
        docker_client,
        solution_code,
        problem["test_code"],
    )

    return {
        "problem_id": problem["problem_id"],
        "difficulty": problem["difficulty"],
        "solution": solution_code[:5000],
        "test_result": result,
        "passed": result["passed"] == result["total"] and result["total"] > 0,
    }


def harness(
    agent_path="./task_agent.py",
    output_dir="./outputs",
    run_id=None,
    num_samples=-1,
    num_workers=3,
):
    dataset = load_dataset(num_samples)
    docker_client = docker.from_env()

    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    output_folder = os.path.join(output_dir, run_id)
    os.makedirs(output_folder, exist_ok=True)

    results = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for problem in dataset:
            futures.append(
                executor.submit(evaluate_problem, problem, agent_path, docker_client)
            )

        for future in futures:
            result = future.result()
            results.append(result)
            print(
                f"  [{result['difficulty']}] {result['problem_id']}: "
                f"{'PASS' if result['passed'] else 'FAIL'}"
            )

    # Save results
    results_path = os.path.join(output_folder, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {results_path}")
    return output_folder


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CodeForge Coding Harness")
    parser.add_argument("--agent_path", type=str, default="./task_agent.py")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=-1)
    parser.add_argument("--num_workers", type=int, default=3)
    args = parser.parse_args()

    harness(
        agent_path=args.agent_path,
        output_dir=args.output_dir,
        run_id=args.run_id,
        num_samples=args.num_samples,
        num_workers=args.num_workers,
    )
