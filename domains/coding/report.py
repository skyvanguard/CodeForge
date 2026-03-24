import argparse
import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))


def report(dname):
    results_path = os.path.join(dname, "results.json")
    if not os.path.exists(results_path):
        print(f"No results.json found in {dname}")
        return None

    with open(results_path, "r") as f:
        results = json.load(f)

    total = len(results)
    if total == 0:
        print("No results to report.")
        return None

    passed = sum(1 for r in results if r.get("passed", False))

    # Scores by difficulty
    by_difficulty = {}
    for r in results:
        diff = r.get("difficulty", "unknown")
        if diff not in by_difficulty:
            by_difficulty[diff] = {"total": 0, "passed": 0}
        by_difficulty[diff]["total"] += 1
        if r.get("passed", False):
            by_difficulty[diff]["passed"] += 1

    test_pass_rate = passed / total if total > 0 else 0.0

    report_data = {
        "test_pass_rate": test_pass_rate,
        "total_problems": total,
        "total_passed": passed,
        "by_difficulty": {
            diff: {
                "total": stats["total"],
                "passed": stats["passed"],
                "pass_rate": stats["passed"] / stats["total"] if stats["total"] > 0 else 0.0,
            }
            for diff, stats in by_difficulty.items()
        },
    }

    # Save report
    report_path = os.path.join(dname, "report.json")
    with open(report_path, "w") as f:
        json.dump(report_data, f, indent=2)

    # Print summary
    print(f"\n{'='*50}")
    print(f"CodeForge Evaluation Report")
    print(f"{'='*50}")
    print(f"Overall Pass Rate: {test_pass_rate:.1%} ({passed}/{total})")
    for diff in ["easy", "medium", "hard"]:
        if diff in report_data["by_difficulty"]:
            stats = report_data["by_difficulty"][diff]
            print(f"  {diff:>8}: {stats['pass_rate']:.1%} ({stats['passed']}/{stats['total']})")
    print(f"{'='*50}\n")

    return report_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CodeForge Report Generator")
    parser.add_argument("--dname", type=str, required=True, help="Directory with results.json")
    args = parser.parse_args()

    report(args.dname)
