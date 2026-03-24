"""Utils for generate_loop.py"""

import fnmatch
import json
import math
import os
import random
import shutil

import numpy as np

from utils.common import read_file
from utils.constants import REPO_NAME
from utils.docker_utils import copy_to_container, log_container_output
from utils.domain_utils import (
    get_domain_score_key,
    get_domain_splits,
    get_domain_stagedeval_frac,
)
from utils.git_utils import commit_repo, get_git_commit_hash


def is_starting_node(genid):
    return genid == "initial" or genid == 0


def get_score(domain, output_dir, genid, split="train"):
    eval_dirname = f"{domain}_eval" if split == "train" else f"{domain}_eval_{split}"
    eval_file = os.path.join(output_dir, f"gen_{genid}/{eval_dirname}/report.json")
    score_key = get_domain_score_key(domain)
    try:
        with open(eval_file, "r") as f:
            eval_results = json.load(f)
        score = eval_results[score_key]
        if math.isnan(score):
            score = None
        return score
    except Exception:
        return None


def get_saved_score(domain, output_dir, genid, split="train", type="agent"):
    agent_score = get_score(domain, output_dir, genid, split=split)

    run_full_eval = get_node_metadata_key(output_dir, genid, "run_full_eval")
    if genid == "initial" or (run_full_eval is None or not run_full_eval):
        stagedeval_frac = get_domain_stagedeval_frac(domain)
        agent_score = agent_score * stagedeval_frac if agent_score is not None else None

    return agent_score


def get_parent_genid(output_dir, genid):
    metadata_file = os.path.join(output_dir, f"gen_{genid}/metadata.json")
    if not os.path.exists(metadata_file):
        return None
    with open(metadata_file, "r") as f:
        metadata = json.load(f)
    return metadata.get("parent_genid", None)


def get_patch_files(output_dir, genid):
    metadata_file = os.path.join(output_dir, f"gen_{genid}/metadata.json")
    if not os.path.exists(metadata_file):
        return []
    with open(metadata_file, "r") as f:
        metadata = json.load(f)
    return metadata.get("prev_patch_files", []) + metadata.get("curr_patch_files", [])


def update_node_metadata(output_dir, genid, data_update):
    metadata_file = os.path.join(output_dir, f"gen_{genid}/metadata.json")
    if not os.path.exists(metadata_file):
        return
    with open(metadata_file, "r") as f:
        metadata = json.load(f)
    metadata.update(data_update)
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=4)


def get_node_metadata_key(output_dir, genid, key):
    metadata_file = os.path.join(output_dir, f"gen_{genid}/metadata.json")
    if not os.path.exists(metadata_file):
        return None
    with open(metadata_file, "r") as f:
        metadata = json.load(f)
    return metadata.get(key, None)


def update_and_save_archive(output_dir, archive, new_node):
    archive.append(new_node)
    archive_file = os.path.join(output_dir, "archive.jsonl")
    with open(archive_file, "a") as f:
        f.write(
            json.dumps({"current_genid": new_node, "archive": archive}) + "\n"
        )
    return archive


def load_archive_data(filepath, last_only=True):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Metadata file not found at {filepath}")
    content = read_file(filepath)
    json_entries = content.split("\n{")
    archive_data = []
    for json_entry in json_entries:
        if not json_entry.startswith("{"):
            json_entry = "{" + json_entry
        metadata = json.loads(json_entry)
        archive_data.append(metadata)
    if last_only:
        return archive_data[-1]
    return archive_data


def get_archive_len(output_dir):
    archive_file = os.path.join(output_dir, "archive.jsonl")
    if not os.path.exists(archive_file):
        return 0
    archive_data = load_archive_data(archive_file, last_only=True)
    return len(archive_data.get("archive", []))


def setup_initial_gen(output_dir, domains, resume=False):
    if resume:
        root_dir = os.path.abspath(os.path.join(output_dir, f"gen_initial/{REPO_NAME}"))
        commit_hash = get_git_commit_hash(root_dir)
        return root_dir, commit_hash

    root_dir = os.path.abspath(os.path.join(output_dir, f"gen_initial/{REPO_NAME}"))
    if os.path.exists(root_dir):
        shutil.rmtree(root_dir)
    os.makedirs(root_dir, exist_ok=True)

    excluded_dirs = {".claude", "outputs", ".git", "__pycache__"}
    excluded_files = {"Dockerfile", "Dockerfile.runner", ".dockerignore", "setup_initial.sh", "LICENSE.md", "CODE_OF_CONDUCT.md", "CONTRIBUTING.md"}
    excluded_patterns = ["venv*", "__pycache__*", "*.png", "outputs_os*", "*.z0*", "*.zip"]

    source_root = os.path.abspath("./")

    def ignore_function(src, names):
        ignored = []
        for name in names:
            full_path = os.path.join(src, name)
            rel = os.path.relpath(full_path, start=source_root).replace(os.sep, "/")
            if any(rel == d or rel.startswith(d.rstrip("/") + "/") for d in excluded_dirs):
                ignored.append(name)
            elif name in excluded_files and os.path.isfile(full_path):
                ignored.append(name)
            else:
                for pattern in excluded_patterns:
                    if fnmatch.fnmatch(name, pattern):
                        ignored.append(name)
                        break
        return ignored

    shutil.copytree("./", root_dir, dirs_exist_ok=True, ignore=ignore_function)

    readme_path = os.path.join(root_dir, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write("""# CodeForge - Self-Improving Coding Agent

This system automatically produces and improves Python coding agents through iterative self-improvement.
The meta-agent analyzes evaluation results and modifies the task agent to achieve higher test pass rates.
""")

    commit_hash = commit_repo(root_dir)
    return root_dir, commit_hash


def filter_patch_by_files(patch_str, target_files):
    lines = patch_str.splitlines()
    filtered_lines = []
    include_block = False
    for line in lines:
        if line.startswith("diff --git"):
            include_block = not any(
                f"a/{target}" in line and f"b/{target}" in line
                for target in target_files
            )
        if include_block:
            filtered_lines.append(line)
    return "\n".join(filtered_lines) + "\n"


def apply_diffs_container(container, patch_files, repo_name=REPO_NAME, verbose=True):
    patch_files = patch_files or []
    for patch_file in patch_files:
        patch_content = read_file(patch_file)
        filtered_patch = filter_patch_by_files(patch_content, ["domains/"])

        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(filtered_patch)
            filtered_patch_file = f.name

        try:
            copy_to_container(container, filtered_patch_file, f"/{repo_name}/parent_patch.txt", verbose=verbose)
            exec_result = container.exec_run(
                f"/bin/sh -c 'patch -p1 < /{repo_name}/parent_patch.txt'",
                workdir=f"/{repo_name}",
            )
            log_container_output(exec_result, verbose=verbose)
            exec_result = container.exec_run(
                f"rm /{repo_name}/parent_patch.txt", workdir=f"/{repo_name}"
            )
            log_container_output(exec_result, verbose=verbose)
        finally:
            os.remove(filtered_patch_file)

    exec_result = container.exec_run("git add --all", workdir=f"/{repo_name}")
    log_container_output(exec_result, verbose=verbose)

    exec_result = container.exec_run("git status --porcelain", workdir=f"/{repo_name}/")
    log_container_output(exec_result, verbose=verbose)
    status_output = exec_result.output.decode("utf-8").strip()

    if status_output:
        exec_result = container.exec_run(
            "git -c user.name='user' -c user.email='you@example.com' commit -m 'apply patches'",
            workdir=f"/{repo_name}/",
        )
        log_container_output(exec_result, verbose=verbose)
        commit_output = exec_result.output.decode("utf-8")
        commit_hash = commit_output.split()[1].strip("[]")
    else:
        exec_result = container.exec_run("git rev-parse HEAD", workdir=f"/{repo_name}/")
        log_container_output(exec_result, verbose=verbose)
        commit_hash = exec_result.output.decode("utf-8").strip()

    return commit_hash


def select_parent(archive, output_dir, domains, method="best"):
    candidates = {}
    for genid in archive:
        valid_parent = (
            get_node_metadata_key(output_dir, genid, "valid_parent")
            if not is_starting_node(genid)
            else True
        )
        if not valid_parent:
            continue
        per_domain_scores = []
        for dom in domains:
            split = "val" if "val" in get_domain_splits(dom) else "train"
            score = get_saved_score(dom, output_dir, genid, split=split, type="agent")
            per_domain_scores.append(score)
        if per_domain_scores and all(s is not None for s in per_domain_scores):
            candidates[genid] = sum(per_domain_scores) / len(per_domain_scores)

    if not candidates:
        candidates[archive[0]] = 0.0

    if method == "random":
        return random.choice(list(candidates.keys()))
    elif method == "latest":
        return list(candidates.keys())[-1]
    elif method == "best":
        return max(candidates, key=candidates.get)
    elif method == "score_prop":
        commits = list(candidates.keys())
        scores = [candidates[c] for c in commits]
        mid_point = np.mean(sorted(scores, reverse=True)[:3])
        scores = [1 / (1 + math.exp(-10 * (s - mid_point))) for s in scores]
        total = sum(scores)
        probs = [s / total for s in scores] if total > 0 else [1 / len(scores)] * len(scores)
        return random.choices(commits, weights=probs)[0]
    else:
        raise ValueError(f"Unknown method '{method}'")


def run_commands_to_check_compilation(container):
    command = ["timeout", "300", "python", "-c", "from meta_agent import MetaAgent"]
    exec_result = container.exec_run(cmd=command, workdir=f"/{REPO_NAME}")
    log_container_output(exec_result)
    if exec_result.exit_code != 0:
        raise Exception("meta_agent is not compilable")

    command = ["timeout", "300", "python", "-c", "from task_agent import TaskAgent"]
    exec_result = container.exec_run(cmd=command, workdir=f"/{REPO_NAME}")
    log_container_output(exec_result)
    if exec_result.exit_code != 0:
        raise Exception("task_agent is not compilable")
