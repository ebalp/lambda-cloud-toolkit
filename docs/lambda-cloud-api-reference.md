# Lambda Cloud API Reference

Quick reference for the Lambda Cloud REST API. Full OpenAPI spec: [`lambda-cloud-openapi.json`](lambda-cloud-openapi.json) (v1.9.3).

**Base URL**: `https://cloud.lambdalabs.com/api/v1`

**Auth**: Bearer token or Basic auth with API key. The key must be in the `LAMBDA_API_KEY` env var, which is sourced from `.sync.env`:
```bash
source .sync.env
curl -u "$LAMBDA_API_KEY:" https://cloud.lambdalabs.com/api/v1/instances
```

Generate API keys at: Lambda Cloud Dashboard → Settings → API Keys.

**Rate limits**: 1 req/s general, 1 req/12s for launch (5/min).

## S3-compatible storage

Lambda provides S3-compatible object storage ("Lambda Filesystem"). Credentials come from env vars sourced from `.sync.env`:
- `LAMBDA_ACCESS_KEY_ID`
- `LAMBDA_SECRET_ACCESS_KEY`

Endpoint: `https://files.{region}.lambda.ai` (e.g., `us-east-2`).

Use the `lambda-gpu sync` CLI for upload/download/ls, or any S3-compatible client (boto3, aws cli with `--endpoint-url`).

## Common endpoints

### Instances

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/instances` | List running instances |
| GET | `/instances/{id}` | Get instance details (id, ip, status, type) |
| POST | `/instances/{id}` | Update instance (rename, tags) |
| POST | `/instance-operations/launch` | Launch new instance(s) |
| POST | `/instance-operations/restart` | Restart instance(s) |
| POST | `/instance-operations/terminate` | Terminate instance(s) |

#### Launch request body
```json
{
  "region_name": "us-east-1",
  "instance_type_name": "gpu_1x_a100",
  "ssh_key_names": ["my-key"],
  "file_system_names": [],
  "name": "my-instance"
}
```

#### Instance statuses
`booting` | `active` | `unhealthy` | `terminated` | `terminating` | `preempted`

### Instance types

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/instance-types` | List all instance types with availability |

Response includes per-type specs (GPUs, vCPUs, RAM, storage), pricing, and `regions_with_capacity_available`.

### GPU instance types

Instance type names follow the pattern: `gpu_{count}x_{gpu_model}[_{variant}]`

To get the current list, query the API:
```bash
curl -u "$LAMBDA_API_KEY:" https://cloud.lambdalabs.com/api/v1/instance-types \
  | python3 -c "import sys,json; [print(f'{k:35} {v[\"instance_type\"][\"description\"]:45} \${v[\"instance_type\"][\"price_cents_per_hour\"]/100:.2f}/hr') for k,v in sorted(json.load(sys.stdin)['data'].items())]"
```

Instance types retrieved from the live API (2026-03-10):

| Instance type | Description | Price |
|---------------|-------------|-------|
| `cpu_4x_general` | 4x CPU General (16 GiB) | $0.20/hr |
| `gpu_1x_a10` | 1x A10 (24 GB PCIe) | $0.86/hr |
| `gpu_1x_a100` | 1x A100 (40 GB PCIe) | $1.48/hr |
| `gpu_1x_a100_sxm4` | 1x A100 (40 GB SXM4) | $1.48/hr |
| `gpu_1x_a6000` | 1x A6000 (48 GB) | $0.92/hr |
| `gpu_1x_b200_sxm6` | 1x B200 (180 GB SXM6) | $6.08/hr |
| `gpu_1x_gh200` | 1x GH200 (96 GB) | $1.99/hr |
| `gpu_1x_h100_pcie` | 1x H100 (80 GB PCIe) | $2.86/hr |
| `gpu_1x_h100_sxm5` | 1x H100 (80 GB SXM5) | $3.78/hr |
| `gpu_1x_rtx6000` | 1x RTX 6000 (24 GB) | $0.58/hr |
| `gpu_2x_a100` | 2x A100 (40 GB PCIe) | $2.96/hr |
| `gpu_2x_a6000` | 2x A6000 (48 GB) | $1.84/hr |
| `gpu_2x_b200_sxm6` | 2x B200 (180 GB SXM6) | $11.94/hr |
| `gpu_2x_h100_sxm5` | 2x H100 (80 GB SXM5) | $7.34/hr |
| `gpu_4x_a100` | 4x A100 (40 GB PCIe) | $5.92/hr |
| `gpu_4x_a6000` | 4x A6000 (48 GB) | $3.68/hr |
| `gpu_4x_b200_sxm6` | 4x B200 (180 GB SXM6) | $23.40/hr |
| `gpu_4x_h100_sxm5` | 4x H100 (80 GB SXM5) | $14.20/hr |
| `gpu_8x_a100` | 8x A100 (40 GB SXM4) | $11.84/hr |
| `gpu_8x_a100_80gb_sxm4` | 8x A100 (80 GB SXM4) | $16.48/hr |
| `gpu_8x_b200_sxm6` | 8x B200 (180 GB SXM6) | $45.92/hr |
| `gpu_8x_h100_sxm5` | 8x H100 (80 GB SXM5) | $27.52/hr |
| `gpu_8x_v100` | 8x Tesla V100 (16 GB) | $5.04/hr |
| `gpu_8x_v100_n` | 8x Tesla V100 (16 GB) | $4.40/hr |

**Note**: Lambda A100 instances are **40 GB** (not 80 GB). The only 80 GB A100 option is `gpu_8x_a100_80gb_sxm4`. Prices and availability change over time — query `/instance-types` for the current list.

### SSH keys

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/ssh-keys` | List SSH keys |
| POST | `/ssh-keys` | Add SSH key (name + public_key) |
| DELETE | `/ssh-keys/{id}` | Delete SSH key |

### Images

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/images` | List available images (OS, CUDA versions) |

### Filesystems

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/file-systems` | List filesystems |
| POST | `/filesystems` | Create filesystem |
| DELETE | `/filesystems/{id}` | Delete filesystem |

### Firewall

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/firewall-rules` | List inbound firewall rules |
| PUT | `/firewall-rules` | Replace inbound firewall rules |

## Key schemas

### InstanceType
```
name: string              # e.g. "gpu_8x_h100_sxm5gdr"
description: string       # e.g. "8x H100 (80 GB SXM5)"
gpu_description: string   # e.g. "H100 (80 GB SXM5)"
price_cents_per_hour: int  # e.g. 3592
specs:
  vcpus: int              # e.g. 208
  memory_gib: int         # e.g. 1800
  storage_gib: int        # e.g. 24780
  gpus: int               # e.g. 8
```

### Instance
```
id: string                # e.g. "0920582c7ff041399e34823a0be62549"
name: string              # user-provided name
ip: string                # public IPv4
private_ip: string        # private IPv4
status: InstanceStatus    # booting|active|unhealthy|terminated|terminating|preempted
ssh_key_names: [string]
file_system_names: [string]
region: { name, description }
instance_type: InstanceType
```

## Error codes

Common error codes in `error.code`:
- `global/invalid-api-key` — bad API key
- `global/insufficient-capacity` — GPU not available (capacity race during launch)
- `global/quota-exceeded` — account quota hit
- `global/account-inactive` — account disabled

## VRAM sizing guide

Rule of thumb for BF16 inference: ~2 GB VRAM per 1B parameters.

| Model size | Min VRAM | Recommended instance |
|------------|----------|---------------------|
| 7–8B | ~16 GB | `gpu_1x_a10` (24 GB) |
| 13B | ~26 GB | `gpu_1x_a100` (40 GB) |
| 27B | ~54 GB | `gpu_1x_gh200` (96 GB) or `gpu_1x_h100_pcie` (80 GB) |
| 70B | ~140 GB | `gpu_8x_a100` (8×40 GB) with tensor parallelism |

**Gotcha**: Lambda `gpu_1x_a100` is **40 GB**, not 80 GB. Models needing >40 GB VRAM (e.g., 27B at BF16) will OOM on a single A100. Use GH200 (96 GB, $1.99/hr) or H100 (80 GB) instead.
