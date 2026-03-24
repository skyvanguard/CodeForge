import argparse
import json
import os
from datetime import datetime

import docker

from utils.constants import REPO_NAME
from utils.docker_utils import (
    build_container,
    cleanup_container,
    copy_from_container,
    copy_to_container,
    log_container_output,
    safe_log,
    setup_logger,
)
from utils.domain_utils import get_domain_splits, get_domain_stagedeval_samples
from utils.gl_utils import (
    apply_diffs_container,
    get_patch_files,
    get_score,
    load_archive_data,
    run_commands_to_check_compilation,
    select_parent,
    setup_initial_gen,
    update_and_save_archive,
    update_node_metadata,
    is_starting_node,
)


def eval_produced_agent(
    container,
    container_output_folder,
    gen_output_dir,
    domain,
    eval_samples=-1,
    eval_workers=3,
):
    splits = get_domain_splits(domain)
    for split in splits:
        safe_log(f"Evaluating the produced agent on {domain} {eval_samples} {split}...")
        eval_run_id = f"{domain}_eval" if split == "train" else f"{domain}_eval_{split}"
        container_evaloutput_folder = os.path.join(container_output_folder, eval_run_id)

        # Run harness
        command = [
            "timeout", "7200",
            "python", "-m", "domains.coding.harness",
            "--agent_path", "./task_agent.py",
            "--output_dir", container_output_folder,
            "--run_id", eval_run_id,
            "--num_samples", str(eval_samples),
            "--num_workers", str(eval_workers),
        ]
        exec_result = container.exec_run(cmd=command, workdir=f"/{REPO_NAME}")
        log_container_output(exec_result)

        # Run report
        command = [
            "timeout", "3600",
            "python", "-m", "domains.coding.report",
            "--dname", os.path.join(container_output_folder, eval_run_id),
        ]
        exec_result = container.exec_run(cmd=command, workdir=f"/{REPO_NAME}")
        log_container_output(exec_result)

        # Copy results to local
        evaloutput_folder = os.path.join(gen_output_dir, eval_run_id)
        copy_from_container(
            container,
            source_path=container_evaloutput_folder,
            dest_path=evaloutput_folder,
        )


def copy_prev_eval_to_container(
    container, prev_eval_path, container_output_folder, current_genid=None
):
    if not os.path.exists(prev_eval_path):
        raise FileNotFoundError(f"Previous eval path not found: {prev_eval_path}")

    prev_eval_path = os.path.normpath(prev_eval_path)
    tail = os.path.join(*prev_eval_path.split(os.sep)[-1:])
    container_prev_eval_path = os.path.join(container_output_folder, tail)

    container.exec_run(["mkdir", "-p", container_output_folder], workdir="/")
    copy_to_container(
        container, source_path=prev_eval_path, dest_path=container_prev_eval_path
    )

    # Prune unnecessary files
    prune_cmds = [
        f"find '{container_prev_eval_path}' -type d -name 'gen_{current_genid}' -prune -exec rm -rf {{}} +",
        f"find '{container_prev_eval_path}' -type d -name '*{REPO_NAME}*' -prune -exec rm -rf {{}} +",
        f"find '{container_prev_eval_path}' -type f -name '*.pyc' -delete",
    ]
    for cmd in prune_cmds:
        container.exec_run(["bash", "-lc", cmd], workdir="/")

    return container_prev_eval_path


def generate(
    docker_client,
    domain,
    output_dir,
    run_id,
    current_genid,
    parent_genid,
    root_dir,
    root_commit="main",
    eval_samples=-1,
    eval_workers=3,
    meta_patch_files=None,
):
    gen_output_dir = os.path.join(output_dir, f"gen_{current_genid}")
    os.makedirs(gen_output_dir, exist_ok=True)

    logger = setup_logger(os.path.join(gen_output_dir, "generate.log"))
    safe_log(f"Generation {current_genid}: parent={parent_genid}")

    prev_patch_files = get_patch_files(output_dir, parent_genid)
    safe_log(f"Parent patch files: {prev_patch_files}")

    # Create Docker container
    image_name = REPO_NAME
    container_name = f"{REPO_NAME}-gen-{current_genid}-{run_id}"
    container = build_container(docker_client, root_dir, image_name, container_name)
    if container is None:
        raise Exception("Failed to create container")
    container.start()
    container_output_folder = "/tmp/codeforge_output/"

    try:
        # Apply parent lineage patches
        commit_hash = apply_diffs_container(container, prev_patch_files)

        # Check compilation
        run_commands_to_check_compilation(container)

        # Copy previous eval results to container
        try:
            container_prev_eval_path = copy_prev_eval_to_container(
                container, output_dir, container_output_folder,
                current_genid=current_genid,
            )
        except FileNotFoundError:
            container_prev_eval_path = container_output_folder

        # Run meta agent in container
        safe_log("Running meta agent...")
        command = [
            "timeout", "3600",
            "python", "run_meta_agent.py",
            "--repo_path", f"/{REPO_NAME}",
            "--evals_folder", container_prev_eval_path,
            "--chat_history_file", os.path.join(container_output_folder, "meta_chat_history.md"),
            "--git_dir", f"/{REPO_NAME}",
            "--base_commit", commit_hash,
            "--outdir", container_output_folder,
            "--iterations_left", str(current_genid),
        ]
        exec_result = container.exec_run(cmd=command, workdir=f"/{REPO_NAME}")
        log_container_output(exec_result)

        # Check compilation after meta agent changes
        try:
            run_commands_to_check_compilation(container)
            valid_parent = True
        except Exception as e:
            safe_log(f"Compilation failed after meta agent: {e}")
            valid_parent = False

        # Copy meta agent outputs
        try:
            patch_file = os.path.join(gen_output_dir, "model_patch.diff")
            copy_from_container(
                container,
                source_path=os.path.join(container_output_folder, "model_patch.diff"),
                dest_path=patch_file,
            )
            curr_patch_files = [patch_file]
        except Exception:
            curr_patch_files = []

        try:
            copy_from_container(
                container,
                source_path=os.path.join(container_output_folder, "meta_chat_history.md"),
                dest_path=os.path.join(gen_output_dir, "meta_chat_history.md"),
            )
        except Exception:
            pass

        # Evaluate the produced agent
        if valid_parent:
            safe_log("Evaluating the produced agent...")
            eval_produced_agent(
                container, container_output_folder, gen_output_dir,
                domain, eval_samples=eval_samples, eval_workers=eval_workers,
            )

        # Save metadata
        metadata = {
            "genid": current_genid,
            "parent_genid": parent_genid,
            "prev_patch_files": prev_patch_files,
            "curr_patch_files": curr_patch_files,
            "valid_parent": valid_parent,
            "run_full_eval": True,
            "timestamp": datetime.now().isoformat(),
        }
        with open(os.path.join(gen_output_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=4)

        # Get score
        score = get_score(domain, output_dir, current_genid)
        safe_log(f"Generation {current_genid} score: {score}")

    except Exception as e:
        safe_log(f"Error in generation {current_genid}: {e}")
        valid_parent = False
        metadata = {
            "genid": current_genid,
            "parent_genid": parent_genid,
            "prev_patch_files": prev_patch_files,
            "curr_patch_files": [],
            "valid_parent": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }
        with open(os.path.join(gen_output_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=4)

    finally:
        # Reset and cleanup
        try:
            container.exec_run(
                cmd=["git", "reset", "--hard", root_commit], workdir=f"/{REPO_NAME}"
            )
            container.exec_run(cmd=["git", "clean", "-fd"], workdir=f"/{REPO_NAME}")
        except Exception:
            pass
        cleanup_container(container)

    return current_genid


def generate_loop(
    run_id=None,
    max_generation=10,
    eval_samples=-1,
    eval_workers=3,
    parent_selection="best",
    resume_from=None,
    domain="coding",
):
    # Setup
    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_dir = os.path.join("./outputs", f"run_{run_id}")
    os.makedirs(output_dir, exist_ok=True)

    logger = setup_logger(os.path.join(output_dir, "generate_loop.log"))
    safe_log(f"Starting CodeForge generate loop: run_id={run_id}")

    docker_client = docker.from_env()

    # Initialize or resume
    if resume_from:
        safe_log(f"Resuming from {resume_from}")
        output_dir = resume_from
        archive_data = load_archive_data(os.path.join(output_dir, "archive.jsonl"))
        archive = archive_data["archive"]
        start_gen = max(g for g in archive if not is_starting_node(g)) + 1 if any(
            not is_starting_node(g) for g in archive
        ) else 1
        root_dir = os.path.abspath(os.path.join(output_dir, f"gen_initial/{REPO_NAME}"))
        from utils.git_utils import get_git_commit_hash
        root_commit = get_git_commit_hash(root_dir)
    else:
        root_dir, root_commit = setup_initial_gen(output_dir, [domain])
        archive = ["initial"]

        # Initial evaluation
        safe_log("Running initial evaluation...")
        image_name = REPO_NAME
        container_name = f"{REPO_NAME}-initial-eval-{run_id}"
        container = build_container(docker_client, root_dir, image_name, container_name)
        if container is None:
            raise Exception("Failed to create initial container")
        container.start()

        try:
            eval_produced_agent(
                container, "/tmp/codeforge_output/",
                os.path.join(output_dir, "gen_initial"),
                domain, eval_samples=eval_samples, eval_workers=eval_workers,
            )
        finally:
            cleanup_container(container)

        initial_score = get_score(domain, output_dir, "initial")
        safe_log(f"Initial score: {initial_score}")

        # Save initial metadata
        initial_metadata = {
            "genid": "initial",
            "valid_parent": True,
            "run_full_eval": True,
            "score": initial_score,
            "timestamp": datetime.now().isoformat(),
        }
        os.makedirs(os.path.join(output_dir, "gen_initial"), exist_ok=True)
        with open(os.path.join(output_dir, "gen_initial/metadata.json"), "w") as f:
            json.dump(initial_metadata, f, indent=4)

        archive = update_and_save_archive(output_dir, [], "initial")
        start_gen = 1

    # Main generation loop
    for gen_id in range(start_gen, start_gen + max_generation):
        safe_log(f"\n{'='*60}")
        safe_log(f"GENERATION {gen_id}")
        safe_log(f"{'='*60}")

        # Select parent
        parent_genid = select_parent(archive, output_dir, [domain], method=parent_selection)
        safe_log(f"Selected parent: {parent_genid}")

        # Generate
        try:
            generate(
                docker_client=docker_client,
                domain=domain,
                output_dir=output_dir,
                run_id=run_id,
                current_genid=gen_id,
                parent_genid=parent_genid,
                root_dir=root_dir,
                root_commit=root_commit,
                eval_samples=eval_samples,
                eval_workers=eval_workers,
            )
        except Exception as e:
            safe_log(f"Generation {gen_id} failed: {e}")

        # Update archive
        archive = update_and_save_archive(output_dir, archive, gen_id)

        # Log progress
        score = get_score(domain, output_dir, gen_id)
        best_score = max(
            (get_score(domain, output_dir, g) or 0.0 for g in archive),
            default=0.0,
        )
        safe_log(f"Gen {gen_id} score: {score}, Best so far: {best_score}")

    safe_log(f"\nCodeForge loop complete. {max_generation} generations produced.")
    safe_log(f"Results in: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CodeForge Self-Improving Loop")
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--max_generation", type=int, default=10)
    parser.add_argument("--eval_samples", type=int, default=-1)
    parser.add_argument("--eval_workers", type=int, default=3)
    parser.add_argument("--parent_selection", type=str, default="best",
                        choices=["best", "random", "latest", "score_prop"])
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--domain", type=str, default="coding")
    args = parser.parse_args()

    generate_loop(
        run_id=args.run_id,
        max_generation=args.max_generation,
        eval_samples=args.eval_samples,
        eval_workers=args.eval_workers,
        parent_selection=args.parent_selection,
        resume_from=args.resume_from,
        domain=args.domain,
    )
