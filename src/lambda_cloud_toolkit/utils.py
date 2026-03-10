"""Shared utilities for lambda_cloud_toolkit."""

import glob as _glob
import os
import re


def find_env_file() -> str | None:
    """Auto-discover a .sync.env file in the current directory.

    Returns the path if found, None otherwise.
    """
    if os.path.isfile(".sync.env"):
        return ".sync.env"
    matches = _glob.glob("*.sync.env")
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        print(f"Multiple .sync.env files found: {', '.join(sorted(matches))}")
        print("Use --env-file to pick one.")
        return None
    return None


def load_env_file(env_file: str) -> None:
    """Source shell export lines from an env file into os.environ.

    Only sets variables that are not already set in the environment,
    so explicit env vars or prior `source` take precedence.
    Handles lines with or without 'export' prefix.
    """
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Match both "export KEY=value" and "KEY=value"
            m = re.match(r'^(?:export\s+)?([A-Za-z_][A-Za-z_0-9]*)=(.*)', line)
            if m:
                key, value = m.group(1), m.group(2)
                value = value.strip().strip('"').strip("'")
                if key not in os.environ and value:
                    os.environ[key] = value
