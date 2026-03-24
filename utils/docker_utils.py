import io
import logging
import os
import tarfile
import threading
import warnings
from pathlib import Path
from typing import Optional, Union

import docker
from docker.models.containers import Container

from utils.constants import REPO_NAME

warnings.filterwarnings(
    "ignore",
    message=r"The default behavior of tarfile extraction has been changed.*",
    category=RuntimeWarning,
)

_thread_local = threading.local()


def get_thread_logger():
    return getattr(_thread_local, "logger", None)


def setup_logger(log_file):
    thread_id = threading.get_ident()
    logger_name = f"docker_logger_{thread_id}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    for handler in logger.handlers:
        logger.removeHandler(handler)

    handler = logging.FileHandler(log_file)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(threadName)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    _thread_local.logger = logger
    return logger


def safe_log(message: str, level: int = logging.INFO, verbose=True):
    if verbose:
        logger = get_thread_logger()
        if logger:
            logger.log(level, message)
        else:
            print(f"[LOG] {message}")


def log_container_output(exec_result, verbose=True):
    if isinstance(exec_result.output, bytes):
        safe_log(f"Container output: {exec_result.output.decode()}", verbose=verbose)
    else:
        for chunk in exec_result.output:
            if chunk:
                safe_log(f"Container output: {chunk.decode().strip()}", verbose=verbose)

    if exec_result.exit_code and exec_result.exit_code != 0:
        error_msg = f"Script failed with exit code {exec_result.exit_code}"
        safe_log(error_msg, logging.ERROR, verbose=verbose)
        raise Exception(error_msg)


def build_container(
    client,
    repo_path="./",
    image_name="app",
    container_name="app-container",
    force_rebuild=False,
    domains=None,
    verbose=True,
):
    try:
        image_exists = any(
            image_name in tag for img in client.images.list() for tag in img.tags
        )
        if force_rebuild or not image_exists:
            safe_log("Building Docker image...", verbose=verbose)
            image, logs = client.images.build(
                path=repo_path,
                tag=image_name,
                rm=True,
                nocache=force_rebuild,
            )
            for log_entry in logs:
                if "stream" in log_entry:
                    safe_log(log_entry["stream"].strip())
            safe_log("Image built successfully.", verbose=verbose)
        else:
            safe_log(
                f"Docker image '{image_name}' already exists. Skipping build.",
                verbose=verbose,
            )

    except Exception as e:
        safe_log(f"Error while building the Docker image: {e}")
        return None

    try:
        try:
            existing = client.containers.get(container_name)
            existing.remove(force=True)
            safe_log(f"Removed existing container '{container_name}'.", verbose=verbose)
        except docker.errors.NotFound:
            pass

        container = client.containers.run(
            image=image_name,
            name=container_name,
            detach=True,
            tty=True,
            stdin_open=True,
            network_mode="host",
            volumes={
                os.path.abspath(repo_path): {"bind": f"/{REPO_NAME}", "mode": "rw"}
            },
            command="tail -f /dev/null",
        )
        safe_log(f"Container '{container_name}' started successfully.", verbose=verbose)
        return container

    except Exception as e:
        safe_log(f"Error while starting the container: {e}", verbose=verbose)
        return None


def copy_to_container(
    container, source_path: Union[str, Path], dest_path: Union[str, Path], verbose=True
) -> None:
    source_path = Path(source_path)
    dest_path = Path(dest_path)

    try:
        if not source_path.exists():
            raise FileNotFoundError(f"Source path not found: {source_path}")

        container_dest_dir = str(dest_path.parent)

        tar_stream = io.BytesIO()
        with tarfile.open(fileobj=tar_stream, mode="w") as tar:
            if source_path.is_file():
                arcname = dest_path.name
                tarinfo = tar.gettarinfo(str(source_path), arcname=arcname)
                tarinfo.uid = 0
                tarinfo.gid = 0
                with open(source_path, "rb") as f:
                    tar.addfile(tarinfo, f)
            else:
                def _reset_uid_gid(ti: tarfile.TarInfo) -> tarfile.TarInfo:
                    ti.uid = 0
                    ti.gid = 0
                    return ti

                tar.add(
                    str(source_path),
                    arcname=dest_path.name,
                    filter=_reset_uid_gid,
                )

        tar_stream.seek(0)
        container.exec_run(f"mkdir -p {container_dest_dir}")
        success = container.put_archive(container_dest_dir, tar_stream)

        if not success:
            raise Exception(f"Failed to copy {source_path} to container")

        safe_log(
            f"Successfully copied {source_path} to container at {dest_path}",
            verbose=verbose,
        )

    except Exception as e:
        safe_log(f"Error copying to container: {e}", logging.ERROR, verbose=verbose)
        raise


def copy_from_container(
    container, source_path: Union[str, Path], dest_path: Union[str, Path], verbose=True
) -> None:
    source_path = Path(source_path)
    dest_path = Path(dest_path)

    try:
        result = container.exec_run(f"test -e {source_path}")
        if result.exit_code and result.exit_code != 0:
            raise FileNotFoundError(
                f"Source path not found in container: {source_path}"
            )

        result = container.exec_run(f"stat -f '%HT' {source_path}")
        is_file = result.output.decode().strip() == "Regular File"

        dest_path.parent.mkdir(parents=True, exist_ok=True)

        bits, stat = container.get_archive(str(source_path))
        archive_data = b"".join(bits)
        stream = io.BytesIO(archive_data)

        with tarfile.open(fileobj=stream, mode="r") as tar:
            if is_file:
                member = tar.getmembers()[0]
                source_file = tar.extractfile(member)
                if source_file is not None:
                    with source_file:
                        data = source_file.read()
                        with open(dest_path, "wb") as dest_file:
                            dest_file.write(data)
            else:
                tar.extractall(path=str(dest_path.parent))
                extracted_path = dest_path.parent / Path(stat["name"]).name
                if extracted_path != dest_path and extracted_path.exists():
                    extracted_path.rename(dest_path)

        safe_log(
            f"Successfully copied from container {source_path} to local path {dest_path}",
            verbose=verbose,
        )

    except Exception as e:
        if verbose:
            safe_log(
                f"Error copying from container: {e}", logging.ERROR, verbose=verbose
            )
        raise


def cleanup_container(container, verbose=True):
    try:
        safe_log(f"Stopping container {container.name}...", verbose=verbose)
        container.stop(timeout=10)
    except Exception as e:
        safe_log(
            f"Error while stopping container {container.name}: {e}",
            level=logging.WARNING,
            verbose=verbose,
        )

    try:
        safe_log(f"Removing container {container.name}...", verbose=verbose)
        container.remove(force=True)
    except Exception as e:
        safe_log(
            f"Error while removing container {container.name}: {e}",
            level=logging.ERROR,
            verbose=verbose,
        )
