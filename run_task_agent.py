import argparse
import os

from task_agent import TaskAgent


def main():
    parser = argparse.ArgumentParser(description='Run CodeForge task agent.')
    parser.add_argument('--problem_id', required=True, help='Problem ID to solve')
    parser.add_argument('--description', required=True, help='Problem description')
    parser.add_argument('--function_signature', required=True, help='Function signature')
    parser.add_argument('--chat_history_file', default='./outputs/chat_history.md', help='Path to chat history file')
    parser.add_argument('--outdir', default='./outputs/', help='Output directory')
    args = parser.parse_args()

    agent = TaskAgent(chat_history_file=args.chat_history_file)
    inputs = {
        "domain": "coding",
        "problem_id": args.problem_id,
        "description": args.description,
        "function_signature": args.function_signature,
    }

    prediction, _ = agent.forward(inputs)

    os.makedirs(args.outdir, exist_ok=True)
    output_file = os.path.join(args.outdir, f"solution_{args.problem_id}.py")
    with open(output_file, "w") as f:
        f.write(str(prediction))
    print(f"Solution saved to {output_file}")


if __name__ == "__main__":
    main()
