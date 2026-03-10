"""Unified CLI for Lambda Cloud Toolkit.

Entry point: `lambda-gpu` (installed via pyproject.toml [project.scripts]).

Subcommands:
    snatch   — Poll for GPU availability, launch instance, optionally bootstrap
    setup    — Bootstrap an existing instance via SSH
    vllm     — Manage vLLM server on an instance
    sync     — Upload/download/list S3-compatible storage
"""

import argparse
import logging
import os
import signal
import sys

import yaml

from lambda_cloud_toolkit.config import LambdaConfig, load_lambda_config
from lambda_cloud_toolkit.instance_setup import bootstrap_instance, _remote_dir_from_url
from lambda_cloud_toolkit.manager import LambdaCloudManager
from lambda_cloud_toolkit.ssh import SSHConnection
from lambda_cloud_toolkit.storage import LambdaStorage
from lambda_cloud_toolkit.utils import find_env_file, load_env_file
from lambda_cloud_toolkit.vllm_server import (
    ensure_vllm_running, install_vllm, start_vllm,
    stop_vllm, vllm_status, wait_for_vllm_ready,
)


def _find_config() -> str | None:
    """Auto-discover config file in current directory."""
    for name in ("lambda-cloud.yaml", "lambda.yaml"):
        if os.path.isfile(name):
            return name
    return None


def _load_raw_config(path: str) -> dict:
    """Load raw YAML config."""
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def _resolve_config(args) -> str:
    """Resolve config path from args or auto-discovery."""
    if hasattr(args, "config") and args.config:
        return args.config
    found = _find_config()
    if not found:
        print("ERROR: No config file found. Use --config or create lambda-cloud.yaml")
        sys.exit(1)
    return found


def _resolve_env_file(args) -> str | None:
    """Resolve env file from args or auto-discovery."""
    if hasattr(args, "env_file") and args.env_file:
        return args.env_file
    return find_env_file()


# ── Subcommand: snatch ──


def cmd_snatch(args):
    """Snatch a GPU instance and optionally bootstrap it."""
    _setup_logging()

    env_file = _resolve_env_file(args)
    if args.setup and not env_file:
        print("ERROR: --setup requires an env file (--env-file or a single *.sync.env)")
        sys.exit(1)
    if env_file:
        load_env_file(env_file)

    config_path = _resolve_config(args)
    raw_config = _load_raw_config(config_path)

    ssh_key_name = raw_config.get("ssh_key_name")
    ssh_key_file = raw_config.get("ssh_key_file", "~/.ssh/id_rsa")
    if not ssh_key_name:
        print("ERROR: ssh_key_name must be set in config")
        sys.exit(1)

    api_key = os.environ.get("LAMBDA_API_KEY")
    if not api_key:
        print("ERROR: LAMBDA_API_KEY not set. Source your .sync.env or use --env-file")
        sys.exit(1)

    target_instances = raw_config.get("instance_preferences", ["gpu_1x_a100", "gpu_1x_a100_sxm4"])

    print("Snatching Lambda instance...")
    print(f"  Targets:       {', '.join(target_instances)}")
    print(f"  SSH key:       {ssh_key_name}")
    print(f"  Poll interval: 10s")
    if args.setup:
        print(f"  Setup:         yes (branch={args.branch})")
    print()

    config = LambdaConfig(
        api_key=api_key,
        ssh_key_name=ssh_key_name,
        model_id="__snatch__",
        instance_type=target_instances[0],
        hf_token=os.environ.get("HF_TOKEN", ""),
        ssh_key_file=ssh_key_file,
        instance_preferences=target_instances,
        poll_interval=10,
    )

    manager = LambdaCloudManager(config)
    try:
        instance = manager.launch()
    except KeyboardInterrupt:
        print("\nStopped.")
        sys.exit(0)
    except RuntimeError as e:
        print(f"\nERROR: {e}")
        sys.exit(1)

    print("\n=== INSTANCE LAUNCHED ===")
    print(f"  ID:     {instance.instance_id}")
    print(f"  IP:     {instance.ip}")
    print(f"  SSH:    ssh -i {ssh_key_file} ubuntu@{instance.ip}")
    print("=========================")

    if args.setup:
        repo_url = raw_config.get("repo_url")
        if not repo_url:
            print("ERROR: repo_url must be set in config for --setup")
            sys.exit(1)
        remote_dir = raw_config.get("repo_dir") or _remote_dir_from_url(repo_url)
        setup_script = raw_config.get("setup_script")

        ssh = SSHConnection(ip=instance.ip, key_file=ssh_key_file)
        print(f"\nWaiting for SSH on {instance.ip}...")
        if not ssh.wait_for_ssh(timeout=300):
            print(f"ERROR: SSH not reachable on {instance.ip} after 300s")
            sys.exit(1)

        print("Bootstrapping instance...")
        bootstrap_instance(
            ssh, env_file_path=env_file, repo_url=repo_url,
            branch=args.branch, remote_dir=remote_dir,
            setup_script=setup_script,
        )
        print(f"\n=== INSTANCE READY ===")
        print(f"  SSH:  ssh -i {ssh_key_file} ubuntu@{instance.ip}")
        print(f"  Repo: {remote_dir}")
        print("======================")


# ── Subcommand: setup ──


def cmd_setup(args):
    """Bootstrap an existing instance."""
    _setup_logging()

    env_file = _resolve_env_file(args)
    if not env_file:
        print("ERROR: No env file found. Use --env-file or place a single *.sync.env in the directory.")
        sys.exit(1)
    load_env_file(env_file)

    config_path = _resolve_config(args)
    raw_config = _load_raw_config(config_path)

    key_file = args.key_file or raw_config.get("ssh_key_file", "~/.ssh/id_rsa")
    ssh = SSHConnection(ip=args.ip, key_file=key_file)

    print(f"Waiting for SSH on {args.ip}...")
    if not ssh.wait_for_ssh(timeout=args.wait_ssh):
        print(f"ERROR: SSH not reachable on {args.ip} after {args.wait_ssh}s")
        sys.exit(1)

    repo_url = raw_config.get("repo_url")
    if not repo_url:
        print("ERROR: repo_url must be set in config")
        sys.exit(1)
    remote_dir = raw_config.get("repo_dir") or _remote_dir_from_url(repo_url)
    setup_script = raw_config.get("setup_script")

    bootstrap_instance(
        ssh, env_file_path=env_file, repo_url=repo_url,
        branch=args.branch, remote_dir=remote_dir,
        setup_script=setup_script,
    )
    print(f"Instance {args.ip} bootstrapped successfully.")


# ── Subcommand: vllm ──


def cmd_vllm(args):
    """Manage vLLM on an instance."""
    _setup_logging()

    config_path = _resolve_config(args)
    raw_config = _load_raw_config(config_path)

    key_file = args.key_file or raw_config.get("ssh_key_file", "~/.ssh/id_rsa")
    ssh = SSHConnection(ip=args.ip, key_file=key_file)

    # ── Status ──
    if args.status:
        info = vllm_status(ssh, port=args.port)
        if info:
            model = info["model"] or "(unknown)"
            print(f"vLLM is running on {args.ip}")
            print(f"  Model:   {model}")
            print(f"  PID:     {info['pid']}")
            print(f"  Command: {info['cmdline']}")
        else:
            print(f"vLLM is not running on {args.ip}")
        return

    # ── Stop ──
    if args.stop:
        info = vllm_status(ssh, port=args.port)
        if info:
            print(f"Stopping vLLM on {args.ip} (model={info.get('model', '?')})...")
            stop_vllm(ssh)
            print("Stopped.")
        else:
            print(f"vLLM is not running on {args.ip}")
        return

    # ── Launch ──
    if not args.model:
        print("ERROR: --model is required when launching (or use --status/--stop)")
        sys.exit(1)

    # Load vllm_args from config if not overridden
    extra_args = args.extra_args
    if extra_args is None:
        gpu_map = raw_config.get("model_gpu_map", {})
        model_cfg = gpu_map.get(args.model) or gpu_map.get("_default", {})
        extra_args = model_cfg.get("vllm_args", "")
        if extra_args:
            print(f"Using vllm_args from config: {extra_args}")

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("ERROR: HF_TOKEN env var not set")
        sys.exit(1)

    if not args.skip_install:
        print(f"Installing vLLM on {args.ip}...")
        install_vllm(ssh, venv_path=args.venv_path)

    print(f"Starting vLLM (model={args.model})...")
    start_vllm(
        ssh, model_id=args.model, hf_token=hf_token,
        port=args.port, extra_args=extra_args,
        venv_path=args.venv_path,
    )

    print(f"Waiting for vLLM to be ready (timeout={args.timeout}s)...")
    if not wait_for_vllm_ready(ssh, port=args.port, timeout=args.timeout):
        print("ERROR: vLLM did not become ready in time")
        sys.exit(1)
    print("vLLM is ready!")

    if args.tunnel:
        print(f"Opening SSH tunnel localhost:{args.local_port} -> {args.ip}:{args.port}")
        tunnel, local_port = ssh.open_tunnel(args.local_port, args.port)
        print(f"Tunnel open. vLLM available at http://localhost:{local_port}/v1")
        print("Press Ctrl+C to stop tunnel and exit.")

        def _cleanup(signum, frame):
            tunnel.terminate()
            tunnel.wait()
            print("\nTunnel closed.")
            sys.exit(0)

        signal.signal(signal.SIGINT, _cleanup)
        signal.signal(signal.SIGTERM, _cleanup)
        tunnel.wait()
    else:
        print(f"vLLM running on {args.ip}:{args.port}")
        print(f"To tunnel: ssh -i {key_file} -N -L {args.port}:localhost:{args.port} ubuntu@{args.ip}")


# ── Subcommand: sync ──


def cmd_sync(args):
    """Sync data with Lambda Cloud Filesystem."""
    _setup_logging()

    env_file = _resolve_env_file(args)
    if env_file:
        load_env_file(env_file)

    config_path = _resolve_config(args)
    raw_config = _load_raw_config(config_path)

    storage_config = raw_config.get("storage", {})
    syncignore = storage_config.get("syncignore", ".syncignore")

    try:
        storage = LambdaStorage.from_config(storage_config, syncignore=syncignore)
    except ValueError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    sync_dir = storage_config.get("sync_dir", "data")

    if args.sync_action == "upload":
        paths = args.paths or [sync_dir]
        for path in paths:
            if not os.path.isdir(path):
                print(f"WARNING: {path} not found, skipping")
                continue
            print(f"Uploading {path}/...")
            result = storage.upload(path, subpath=path)
            if result.stdout:
                print(result.stdout)
        print("Upload complete.")

    elif args.sync_action == "download":
        paths = args.paths or [sync_dir]
        for path in paths:
            print(f"Downloading {path}/...")
            result = storage.download(path, subpath=path)
            if result.stdout:
                print(result.stdout)
        print("Download complete.")

    elif args.sync_action == "ls":
        path = args.paths[0] if args.paths else None
        result = storage.ls(path)
        if result.stdout:
            print(result.stdout, end="")
        else:
            print("(empty)")


# ── Main parser ──


def main():
    parser = argparse.ArgumentParser(
        prog="lambda-gpu",
        description="Lambda Cloud GPU instance management and vLLM deployment",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ── snatch ──
    p_snatch = subparsers.add_parser("snatch", help="Poll for GPU availability and launch")
    p_snatch.add_argument("--config", help="Path to lambda-cloud.yaml")
    p_snatch.add_argument("--env-file", help="Path to .sync.env file")
    p_snatch.add_argument("--setup", action="store_true", help="Bootstrap after launch")
    p_snatch.add_argument("--branch", default="main", help="Git branch for --setup")
    p_snatch.set_defaults(func=cmd_snatch)

    # ── setup ──
    p_setup = subparsers.add_parser("setup", help="Bootstrap an existing instance")
    p_setup.add_argument("--ip", required=True, help="Instance IP address")
    p_setup.add_argument("--config", help="Path to lambda-cloud.yaml")
    p_setup.add_argument("--env-file", help="Path to .sync.env file")
    p_setup.add_argument("--key-file", help="SSH key file")
    p_setup.add_argument("--branch", default="main", help="Git branch")
    p_setup.add_argument("--wait-ssh", type=int, default=300, help="SSH timeout (seconds)")
    p_setup.set_defaults(func=cmd_setup)

    # ── vllm ──
    p_vllm = subparsers.add_parser("vllm", help="Manage vLLM on an instance")
    p_vllm.add_argument("--ip", required=True, help="Instance IP address")
    p_vllm.add_argument("--config", help="Path to lambda-cloud.yaml")
    p_vllm.add_argument("--key-file", help="SSH key file")
    p_vllm.add_argument("--port", type=int, default=8000, help="vLLM port")
    p_vllm.add_argument("--model", help="Model ID to serve")
    p_vllm.add_argument("--status", action="store_true", help="Check status")
    p_vllm.add_argument("--stop", action="store_true", help="Stop vLLM")
    p_vllm.add_argument("--venv-path", default="/home/ubuntu/vllm-venv", help="Remote venv path")
    p_vllm.add_argument("--extra-args", default=None, help="Extra vLLM args")
    p_vllm.add_argument("--tunnel", action="store_true", help="Open SSH tunnel")
    p_vllm.add_argument("--local-port", type=int, default=8000, help="Local tunnel port")
    p_vllm.add_argument("--timeout", type=int, default=900, help="Readiness timeout")
    p_vllm.add_argument("--skip-install", action="store_true", help="Skip vLLM install")
    p_vllm.set_defaults(func=cmd_vllm)

    # ── sync ──
    p_sync = subparsers.add_parser("sync", help="Sync data with Lambda Filesystem")
    sync_sub = p_sync.add_subparsers(dest="sync_action", required=True)

    for action in ("upload", "download", "ls"):
        p = sync_sub.add_parser(action, help=f"{action.capitalize()} data")
        p.add_argument("paths", nargs="*", help="Paths to sync")
        p.add_argument("--config", help="Path to lambda-cloud.yaml")
        p.add_argument("--env-file", help="Path to .sync.env file")
        p.set_defaults(func=cmd_sync)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
