"""Tests for lambda_cloud_toolkit.cli — CLI argument parsing."""

import subprocess
import sys
from unittest.mock import patch

import pytest


class TestCLIHelp:
    def test_main_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "lambda_cloud_toolkit.cli", "--help"],
            capture_output=True, text=True, cwd="/Users/enrique/lambda-cloud-toolkit",
        )
        assert result.returncode == 0
        assert "lambda-gpu" in result.stdout or "Lambda Cloud" in result.stdout

    def test_snatch_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "lambda_cloud_toolkit.cli", "snatch", "--help"],
            capture_output=True, text=True, cwd="/Users/enrique/lambda-cloud-toolkit",
        )
        assert result.returncode == 0
        assert "--setup" in result.stdout

    def test_setup_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "lambda_cloud_toolkit.cli", "setup", "--help"],
            capture_output=True, text=True, cwd="/Users/enrique/lambda-cloud-toolkit",
        )
        assert result.returncode == 0
        assert "--ip" in result.stdout

    def test_vllm_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "lambda_cloud_toolkit.cli", "vllm", "--help"],
            capture_output=True, text=True, cwd="/Users/enrique/lambda-cloud-toolkit",
        )
        assert result.returncode == 0
        assert "--model" in result.stdout

    def test_sync_upload_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "lambda_cloud_toolkit.cli", "sync", "upload", "--help"],
            capture_output=True, text=True, cwd="/Users/enrique/lambda-cloud-toolkit",
        )
        assert result.returncode == 0
        assert "paths" in result.stdout or "upload" in result.stdout.lower()

    def test_sync_download_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "lambda_cloud_toolkit.cli", "sync", "download", "--help"],
            capture_output=True, text=True, cwd="/Users/enrique/lambda-cloud-toolkit",
        )
        assert result.returncode == 0

    def test_sync_ls_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "lambda_cloud_toolkit.cli", "sync", "ls", "--help"],
            capture_output=True, text=True, cwd="/Users/enrique/lambda-cloud-toolkit",
        )
        assert result.returncode == 0

    def test_no_command_shows_usage(self):
        result = subprocess.run(
            [sys.executable, "-m", "lambda_cloud_toolkit.cli"],
            capture_output=True, text=True, cwd="/Users/enrique/lambda-cloud-toolkit",
        )
        assert result.returncode != 0  # should fail without a command


class TestConfigDiscovery:
    def test_find_config_lambda_cloud_yaml(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "lambda-cloud.yaml").write_text("ssh_key_name: test\n")
        from lambda_cloud_toolkit.cli import _find_config
        assert _find_config() == "lambda-cloud.yaml"

    def test_find_config_lambda_yaml(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "lambda.yaml").write_text("ssh_key_name: test\n")
        from lambda_cloud_toolkit.cli import _find_config
        assert _find_config() == "lambda.yaml"

    def test_find_config_prefers_lambda_cloud_yaml(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "lambda-cloud.yaml").write_text("ssh_key_name: test1\n")
        (tmp_path / "lambda.yaml").write_text("ssh_key_name: test2\n")
        from lambda_cloud_toolkit.cli import _find_config
        assert _find_config() == "lambda-cloud.yaml"

    def test_find_config_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from lambda_cloud_toolkit.cli import _find_config
        assert _find_config() is None
