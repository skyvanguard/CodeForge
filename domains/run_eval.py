import argparse
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from domains.coding.harness import harness
from domains.coding.report import report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CodeForge Evaluation Runner")
    parser.add_argument("--agent_path", type=str, default="./task_agent.py")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--num_samples", type=int, default=-1)
    parser.add_argument("--num_workers", type=int, default=3)
    args = parser.parse_args()

    output_folder = harness(
        agent_path=args.agent_path,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        num_workers=args.num_workers,
    )

    report(dname=output_folder)
