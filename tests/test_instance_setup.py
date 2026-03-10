"""Tests for lambda_cloud_toolkit.instance_setup."""

import subprocess
from unittest.mock import MagicMock, call, patch

import pytest

from lambda_cloud_toolkit.ssh import SSHConnection
from lambda_cloud_toolkit.instance_setup import bootstrap_instance, _remote_dir_from_url, setup_github_credentials


@pytest.fixture
def ssh():
    """Create a mock SSHConnection."""
    mock = MagicMock(spec=SSHConnection)
    mock.ip = "10.0.0.1"
    return mock


class TestRemoteDirFromUrl:
    def test_https_url(self):
        assert _remote_dir_from_url("https://github.com/org/my-repo.git") == "/home/ubuntu/my-repo"

    def test_https_url_no_git_suffix(self):
        assert _remote_dir_from_url("https://github.com/org/my-repo") == "/home/ubuntu/my-repo"

    def test_ssh_url(self):
        assert _remote_dir_from_url("git@github.com:org/my-repo.git") == "/home/ubuntu/my-repo"


class TestSetupGithubCredentials:
    def test_uploads_env_file(self, ssh):
        setup_github_credentials(ssh, "/local/.sync.env")
        ssh.upload_file.assert_called_once_with("/local/.sync.env", "/home/ubuntu/.sync.env")

    def test_configures_git_credential(self, ssh):
        setup_github_credentials(ssh, "/local/.sync.env")
        # Second call should be the git config command
        assert ssh.run.call_count == 1
        cmd = ssh.run.call_args[0][0]
        assert "GITHUB_TOKEN" in cmd
        assert "git config" in cmd


class TestBootstrapInstance:
    def test_runs_all_steps_in_order(self, ssh, tmp_path):
        env_file = tmp_path / ".sync.env"
        env_file.write_text("export HF_TOKEN=test\n")

        bootstrap_instance(
            ssh,
            env_file_path=str(env_file),
            repo_url="https://github.com/org/repo.git",
            branch="main",
            install_claude=False,
        )

        # Verify the key steps were called
        run_calls = [c[0][0] for c in ssh.run.call_args_list]

        # Should have: github creds, clone, mv env, git identity, uv install, uv sync, bashrc
        assert any("git config" in c and "GITHUB_TOKEN" in c for c in run_calls), "Missing github credential setup"
        assert any("git clone" in c for c in run_calls), "Missing git clone"
        assert any("mv" in c and ".sync.env" in c for c in run_calls), "Missing env file move"
        assert any("GIT_USER_NAME" in c for c in run_calls), "Missing git identity config"
        assert any("uv" in c and "install.sh" in c for c in run_calls), "Missing uv installation"
        assert any("uv sync" in c for c in run_calls), "Missing uv sync"
        assert any(".bashrc" in c for c in run_calls), "Missing bashrc setup"

    def test_skips_setup_script_when_none(self, ssh, tmp_path):
        env_file = tmp_path / ".sync.env"
        env_file.write_text("export HF_TOKEN=test\n")

        bootstrap_instance(
            ssh,
            env_file_path=str(env_file),
            repo_url="https://github.com/org/repo.git",
            setup_script=None,
            install_claude=False,
        )

        run_calls = [c[0][0] for c in ssh.run.call_args_list]
        assert not any("lambda-sync" in c for c in run_calls)

    def test_runs_setup_script_when_provided(self, ssh, tmp_path):
        env_file = tmp_path / ".sync.env"
        env_file.write_text("export HF_TOKEN=test\n")

        bootstrap_instance(
            ssh,
            env_file_path=str(env_file),
            repo_url="https://github.com/org/repo.git",
            setup_script="my-setup.sh",
            install_claude=False,
        )

        run_calls = [c[0][0] for c in ssh.run.call_args_list]
        assert any("my-setup.sh" in c for c in run_calls)

    def test_uses_custom_remote_dir(self, ssh, tmp_path):
        env_file = tmp_path / ".sync.env"
        env_file.write_text("export HF_TOKEN=test\n")

        bootstrap_instance(
            ssh,
            env_file_path=str(env_file),
            repo_url="https://github.com/org/repo.git",
            remote_dir="/home/ubuntu/custom-dir",
            install_claude=False,
        )

        run_calls = [c[0][0] for c in ssh.run.call_args_list]
        assert any("/home/ubuntu/custom-dir" in c for c in run_calls)

    def test_installs_claude_when_enabled(self, ssh, tmp_path):
        env_file = tmp_path / ".sync.env"
        env_file.write_text("export HF_TOKEN=test\n")

        bootstrap_instance(
            ssh,
            env_file_path=str(env_file),
            repo_url="https://github.com/org/repo.git",
            install_claude=True,
        )

        run_calls = [c[0][0] for c in ssh.run.call_args_list]
        assert any("claude" in c for c in run_calls)

    def test_skips_claude_when_disabled(self, ssh, tmp_path):
        env_file = tmp_path / ".sync.env"
        env_file.write_text("export HF_TOKEN=test\n")

        bootstrap_instance(
            ssh,
            env_file_path=str(env_file),
            repo_url="https://github.com/org/repo.git",
            install_claude=False,
        )

        run_calls = [c[0][0] for c in ssh.run.call_args_list]
        assert not any("claude.ai/install" in c for c in run_calls)

    def test_derives_remote_dir_from_url(self, ssh, tmp_path):
        env_file = tmp_path / ".sync.env"
        env_file.write_text("export HF_TOKEN=test\n")

        bootstrap_instance(
            ssh,
            env_file_path=str(env_file),
            repo_url="https://github.com/org/my-project.git",
            install_claude=False,
        )

        run_calls = [c[0][0] for c in ssh.run.call_args_list]
        assert any("/home/ubuntu/my-project" in c for c in run_calls)
