# lambda-cloud-toolkit

Python toolkit for managing Lambda Cloud GPU instances: snatch available GPUs, bootstrap instances, deploy vLLM, and sync data via S3-compatible storage.

## Features

- **GPU snatching** -- poll Lambda Cloud for GPU availability across preferred instance types and auto-launch when capacity appears
- **Instance bootstrapping** -- clone repos, upload credentials, install uv/Python, and run setup scripts over SSH
- **vLLM deployment** -- install, start, stop, and monitor vLLM servers on remote instances with SSH-proxied health checks
- **SSH tunneling** -- forward remote vLLM ports to localhost with automatic port conflict resolution
- **S3 data sync** -- upload, download, and list files on Lambda Cloud Filesystem (S3-compatible)
- **Safety nets** -- context manager, atexit, and signal handlers ensure instances are terminated even on crashes
- **No paramiko** -- all SSH operations use subprocess calls to the system `ssh`/`scp` binaries

## Installation

```bash
# With pip
pip install git+https://github.com/ebalp/lambda-cloud-toolkit.git

# With uv
uv add lambda-cloud-toolkit --git https://github.com/ebalp/lambda-cloud-toolkit.git
```

Requires Python 3.12+.

## Quick Start

```bash
# 1. Copy and fill in your credentials
cp examples/env.template .sync.env
cp examples/lambda-cloud.yaml.example lambda-cloud.yaml
# Edit both files with your values

# 2. Source credentials
source .sync.env

# 3. Snatch a GPU and bootstrap the instance
lambda-gpu snatch --setup
```

## CLI Reference

The `lambda-gpu` command is the main entry point. It auto-discovers `lambda-cloud.yaml` in the current directory.

### `lambda-gpu snatch`

Poll for GPU availability across preferred instance types and launch when capacity appears.

```bash
# Basic snatch (just launch, no setup)
lambda-gpu snatch

# Snatch and bootstrap the instance
lambda-gpu snatch --setup

# Snatch, bootstrap, and check out a specific branch
lambda-gpu snatch --setup --branch feature-branch

# Use a specific config and env file
lambda-gpu snatch --config path/to/lambda-cloud.yaml --env-file my.sync.env
```

| Flag | Default | Description |
|------|---------|-------------|
| `--config` | auto-discover | Path to `lambda-cloud.yaml` |
| `--env-file` | auto-discover | Path to `.sync.env` file |
| `--setup` | off | Bootstrap the instance after launch |
| `--branch` | `main` | Git branch for `--setup` |

### `lambda-gpu setup`

Bootstrap an existing instance (upload credentials, clone repo, install dependencies).

```bash
lambda-gpu setup --ip 192.0.2.10

# With a specific SSH key and branch
lambda-gpu setup --ip 192.0.2.10 --key-file ~/.ssh/lambda.pem --branch dev
```

| Flag | Default | Description |
|------|---------|-------------|
| `--ip` | required | Instance IP address |
| `--config` | auto-discover | Path to `lambda-cloud.yaml` |
| `--env-file` | auto-discover | Path to `.sync.env` file |
| `--key-file` | from config | SSH private key file |
| `--branch` | `main` | Git branch to check out |
| `--wait-ssh` | `300` | SSH reachability timeout in seconds |

### `lambda-gpu vllm`

Manage the vLLM server on a remote instance.

```bash
# Launch vLLM with a model and open an SSH tunnel
lambda-gpu vllm --ip 192.0.2.10 --model meta-llama/Llama-3.1-8B-Instruct --tunnel

# Check status
lambda-gpu vllm --ip 192.0.2.10 --status

# Stop the server
lambda-gpu vllm --ip 192.0.2.10 --stop

# Launch with extra vLLM args (overrides config)
lambda-gpu vllm --ip 192.0.2.10 --model meta-llama/Llama-3.1-8B-Instruct \
  --extra-args "--max-model-len 4096 --enforce-eager"

# Skip vLLM installation (already installed)
lambda-gpu vllm --ip 192.0.2.10 --model meta-llama/Llama-3.1-8B-Instruct --skip-install
```

| Flag | Default | Description |
|------|---------|-------------|
| `--ip` | required | Instance IP address |
| `--model` | required (for launch) | HuggingFace model ID |
| `--status` | off | Check if vLLM is running |
| `--stop` | off | Stop the vLLM server |
| `--tunnel` | off | Open SSH tunnel after launch |
| `--local-port` | `8000` | Local port for tunnel |
| `--port` | `8000` | Remote vLLM port |
| `--venv-path` | `/home/ubuntu/vllm-venv` | Remote virtualenv path |
| `--extra-args` | from config | Extra args passed to `vllm serve` |
| `--timeout` | `900` | Readiness timeout in seconds |
| `--skip-install` | off | Skip `pip install vllm` step |
| `--config` | auto-discover | Path to `lambda-cloud.yaml` |
| `--key-file` | from config | SSH private key file |

### `lambda-gpu sync`

Upload, download, or list files on Lambda Cloud Filesystem (S3-compatible storage).

```bash
# Upload default sync directory
lambda-gpu sync upload

# Upload specific paths
lambda-gpu sync upload data/results data/logs

# Download default sync directory
lambda-gpu sync download

# Download specific paths
lambda-gpu sync download data/results

# List bucket contents
lambda-gpu sync ls
lambda-gpu sync ls data/results
```

| Flag | Default | Description |
|------|---------|-------------|
| `paths` | `sync_dir` from config | Paths to sync (positional) |
| `--config` | auto-discover | Path to `lambda-cloud.yaml` |
| `--env-file` | auto-discover | Path to `.sync.env` file |

## Python API

### LambdaCloudManager

Manage instance lifecycle programmatically with automatic termination safety nets.

```python
from lambda_cloud_toolkit import LambdaCloudManager, load_lambda_config

config = load_lambda_config("lambda-cloud.yaml", model_id="meta-llama/Llama-3.1-8B-Instruct")

# As a context manager (auto-terminates on exit)
with LambdaCloudManager(config) as manager:
    print(f"Instance IP: {manager.instance.ip}")
    print(f"Instance ID: {manager.instance.instance_id}")
    print(f"vLLM URL: {manager.get_base_url()}")
    # ... run experiments ...
# Instance is terminated automatically

# Manual lifecycle
manager = LambdaCloudManager(config)
instance = manager.launch()  # polls until GPU is available
# ... use instance ...
manager.terminate()

# List available GPU types
available = manager.list_available()
for gpu in available:
    print(f"{gpu['name']}: {gpu['description']} ({gpu['available_regions']})")
```

### SSHConnection

Run commands, upload files, and open tunnels over SSH.

```python
from lambda_cloud_toolkit.ssh import SSHConnection

ssh = SSHConnection(ip="192.0.2.10", key_file="~/.ssh/lambda.pem")

# Wait for instance to be reachable
ssh.wait_for_ssh(timeout=300)

# Run commands
result = ssh.run("nvidia-smi", timeout=30)
print(result.stdout)

# Run in background (survives SSH disconnect)
ssh.run_background("python train.py")

# Upload a file
ssh.upload_file("local_data.csv", "/home/ubuntu/data.csv")

# Open an SSH tunnel
tunnel, local_port = ssh.open_tunnel(local_port=8000, remote_port=8000)
print(f"Tunnel on localhost:{local_port}")
# ... use tunnel ...
tunnel.terminate()
```

### vLLM Server Functions

```python
from lambda_cloud_toolkit.ssh import SSHConnection
from lambda_cloud_toolkit.vllm_server import (
    ensure_vllm_running, install_vllm, start_vllm,
    stop_vllm, vllm_status, wait_for_vllm_ready,
)

ssh = SSHConnection(ip="192.0.2.10")

# All-in-one: install + start + wait (skips if already running)
ensure_vllm_running(ssh, model_id="meta-llama/Llama-3.1-8B-Instruct", hf_token="hf_...")

# Or step by step
install_vllm(ssh)
start_vllm(ssh, model_id="meta-llama/Llama-3.1-8B-Instruct", hf_token="hf_...",
            extra_args="--max-model-len 4096")
wait_for_vllm_ready(ssh, timeout=900)

# Check status
status = vllm_status(ssh)
if status:
    print(f"Model: {status['model']}, PID: {status['pid']}")

# Stop
stop_vllm(ssh)
```

### LambdaStorage

Sync data with Lambda Cloud Filesystem.

```python
from lambda_cloud_toolkit.storage import LambdaStorage

storage = LambdaStorage(
    bucket_name="your-bucket-uuid",
    access_key_id="...",
    secret_access_key="...",
)

# Upload a directory
storage.upload("data/results", subpath="data/results")

# Download
storage.download("data/results", subpath="data/results")

# List contents
result = storage.ls("data/")
print(result.stdout)
```

### Instance Bootstrap

```python
from lambda_cloud_toolkit.ssh import SSHConnection
from lambda_cloud_toolkit.instance_setup import bootstrap_instance

ssh = SSHConnection(ip="192.0.2.10")
ssh.wait_for_ssh(timeout=300)

bootstrap_instance(
    ssh,
    env_file_path=".sync.env",
    repo_url="https://github.com/your-org/your-repo.git",
    branch="main",
    setup_script="scripts/setup.sh",  # optional post-clone script
)
```

## Configuration

### `lambda-cloud.yaml`

```yaml
# SSH key registered in Lambda Cloud
ssh_key_name: "my-lambda-key"
ssh_key_file: "~/.ssh/my-lambda-key.pem"

# Repo to clone during bootstrap (used by snatch --setup and setup)
repo_url: "https://github.com/your-org/your-repo.git"
# repo_dir: "/home/ubuntu/your-repo"  # derived from repo_url if omitted
# setup_script: "scripts/setup.sh"    # run after clone (optional)

defaults:
  region: us-east-1
  vllm_port: 8000
  vllm_venv_path: /home/ubuntu/vllm-venv
  max_launch_retries: 5
  launch_retry_delay: 60
  readiness_timeout: 900
  concurrent_per_model: 10

# GPU types to try when snatching (in priority order)
instance_preferences:
  - gpu_1x_a100
  - gpu_1x_a100_sxm4

# Per-model overrides
model_gpu_map:
  meta-llama/Llama-3.1-8B-Instruct:
    instance_type: gpu_1x_a10
    instance_preferences:
      - gpu_1x_a10
      - gpu_1x_a100
    vllm_args: "--max-model-len 4096"
  _default:
    instance_type: gpu_1x_a100
    vllm_args: "--max-model-len 4096"

# S3-compatible storage (for lambda-gpu sync)
storage:
  bucket_name: "<uuid>"  # or set BUCKET_NAME env var
  endpoint_url: "https://files.us-east-2.lambda.ai"
  region: us-east-2
  sync_dir: data          # default directory for upload/download
  syncignore: .syncignore # exclusion patterns file
```

### `.syncignore`

Works like `.gitignore` -- one glob pattern per line, used as `--exclude` flags for `aws s3 sync`:

```
__pycache__/*
*.pyc
.venv/*
```

## Credentials

All credentials are read from environment variables. Use a `.sync.env` file to manage them:

```bash
# Lambda Cloud API key (instance management)
export LAMBDA_API_KEY=<your-key>

# HuggingFace token (gated model downloads)
export HF_TOKEN=<your-token>

# Lambda Filesystem S3 credentials (data sync)
export BUCKET_NAME=<filesystem-bucket-uuid>
export LAMBDA_ACCESS_KEY_ID=<key>
export LAMBDA_SECRET_ACCESS_KEY=<secret>
export LAMBDA_REGION=us-east-2
export LAMBDA_ENDPOINT_URL=https://files.us-east-2.lambda.ai

# GitHub token (clone private repos during bootstrap)
export GITHUB_TOKEN=<personal-access-token>

# Git identity (configured on instance during bootstrap)
export GIT_USER_NAME="Your Name"
export GIT_USER_EMAIL=you@example.com
```

The CLI auto-discovers `.sync.env` or `*.sync.env` in the current directory. Override with `--env-file`.

See `examples/env.template` for a complete template.

## Development

```bash
git clone https://github.com/ebalp/lambda-cloud-toolkit.git
cd lambda-cloud-toolkit
uv sync

# Run unit tests
uv run pytest

# Run live tests (requires a running Lambda Cloud instance)
uv run pytest -m live
```
