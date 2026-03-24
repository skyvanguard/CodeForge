import argparse
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from domains.coding.report import report as report_coding


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CodeForge Report Generator")
    parser.add_argument("--dname", required=True, help="Path to harness outputs")
    parser.add_argument("--domain", type=str, default="coding", choices=["coding"])
    args = parser.parse_args()

    if args.domain == "coding":
        report_coding(dname=args.dname)
