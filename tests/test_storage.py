"""Tests for lambda_cloud_toolkit.storage — LambdaStorage."""

import os
import subprocess
from unittest.mock import patch, MagicMock

import pytest

from lambda_cloud_toolkit.storage import LambdaStorage


@pytest.fixture
def storage(tmp_path):
    """Create a LambdaStorage instance with test credentials."""
    return LambdaStorage(
        bucket_name="test-bucket-uuid",
        endpoint_url="https://files.test.lambda.ai",
        region="us-east-2",
        access_key_id="test-key-id",
        secret_access_key="test-secret-key",
    )


class TestInit:
    def test_requires_bucket_name(self):
        with pytest.raises(ValueError, match="bucket_name"):
            LambdaStorage(
                access_key_id="k", secret_access_key="s",
            )

    def test_requires_credentials(self):
        with pytest.raises(ValueError, match="credentials"):
            LambdaStorage(bucket_name="b")

    def test_reads_from_env(self):
        env = {
            "BUCKET_NAME": "env-bucket",
            "LAMBDA_ACCESS_KEY_ID": "env-key",
            "LAMBDA_SECRET_ACCESS_KEY": "env-secret",
            "LAMBDA_ENDPOINT_URL": "https://env.endpoint",
            "LAMBDA_REGION": "us-west-1",
        }
        with patch.dict(os.environ, env, clear=False):
            s = LambdaStorage()
            assert s.bucket_name == "env-bucket"
            assert s.endpoint_url == "https://env.endpoint"
            assert s.region == "us-west-1"

    def test_explicit_overrides_env(self):
        env = {"BUCKET_NAME": "env-bucket", "LAMBDA_ACCESS_KEY_ID": "env-k", "LAMBDA_SECRET_ACCESS_KEY": "env-s"}
        with patch.dict(os.environ, env, clear=False):
            s = LambdaStorage(bucket_name="explicit-bucket")
            assert s.bucket_name == "explicit-bucket"


class TestEnv:
    def test_sets_aws_credentials(self, storage):
        env = storage._env()
        assert env["AWS_ACCESS_KEY_ID"] == "test-key-id"
        assert env["AWS_SECRET_ACCESS_KEY"] == "test-secret-key"
        assert env["AWS_DEFAULT_REGION"] == "us-east-2"

    def test_sets_checksum_workaround(self, storage):
        env = storage._env()
        assert env["AWS_REQUEST_CHECKSUM_CALCULATION"] == "when_required"
        assert env["AWS_RESPONSE_CHECKSUM_VALIDATION"] == "when_required"


class TestExcludeFlags:
    def test_no_syncignore(self, storage):
        assert storage._exclude_flags() == []

    def test_parses_syncignore(self, tmp_path, storage):
        ignore_file = tmp_path / ".syncignore"
        ignore_file.write_text("# comment\n\n.DS_Store\n*.pyc\n")
        storage._syncignore = str(ignore_file)
        flags = storage._exclude_flags()
        assert flags == ["--exclude", ".DS_Store", "--exclude", "*.pyc"]


class TestBucketUri:
    def test_base_uri(self, storage):
        assert storage._bucket_uri() == "s3://test-bucket-uuid/"

    def test_with_subpath(self, storage):
        assert storage._bucket_uri("data/results") == "s3://test-bucket-uuid/data/results/"

    def test_strips_slashes(self, storage):
        assert storage._bucket_uri("/data/results/") == "s3://test-bucket-uuid/data/results/"


class TestUpload:
    @patch("lambda_cloud_toolkit.storage.subprocess.run")
    def test_upload_calls_aws_sync(self, mock_run, storage, tmp_path):
        local_dir = tmp_path / "data"
        local_dir.mkdir()
        mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

        storage.upload(str(local_dir), subpath="phase0/data")

        cmd = mock_run.call_args[0][0]
        assert cmd[0:3] == ["aws", "s3", "sync"]
        assert str(local_dir) + "/" in cmd
        assert "s3://test-bucket-uuid/phase0/data/" in cmd
        assert "--endpoint-url" in cmd

    def test_raises_if_dir_missing(self, storage):
        with pytest.raises(FileNotFoundError):
            storage.upload("/nonexistent/dir")

    @patch("lambda_cloud_toolkit.storage.subprocess.run")
    def test_upload_includes_exclude_flags(self, mock_run, storage, tmp_path):
        local_dir = tmp_path / "data"
        local_dir.mkdir()
        ignore = tmp_path / ".syncignore"
        ignore.write_text("*.pyc\n")
        storage._syncignore = str(ignore)
        mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

        storage.upload(str(local_dir))
        cmd = mock_run.call_args[0][0]
        assert "--exclude" in cmd
        assert "*.pyc" in cmd


class TestDownload:
    @patch("lambda_cloud_toolkit.storage.subprocess.run")
    def test_download_calls_aws_sync(self, mock_run, storage, tmp_path):
        local_dir = tmp_path / "data"
        mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

        storage.download(str(local_dir), subpath="phase0/data")

        cmd = mock_run.call_args[0][0]
        assert cmd[0:3] == ["aws", "s3", "sync"]
        assert "s3://test-bucket-uuid/phase0/data/" in cmd
        assert str(local_dir) + "/" in cmd

    @patch("lambda_cloud_toolkit.storage.subprocess.run")
    def test_download_creates_dir(self, mock_run, storage, tmp_path):
        local_dir = tmp_path / "new" / "nested" / "dir"
        mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

        storage.download(str(local_dir))
        assert local_dir.exists()


class TestLs:
    @patch("lambda_cloud_toolkit.storage.subprocess.run")
    def test_ls_calls_aws_ls(self, mock_run, storage):
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="2024-01-01 data/\n", stderr=""
        )

        result = storage.ls("data")
        cmd = mock_run.call_args[0][0]
        assert cmd[0:3] == ["aws", "s3", "ls"]
        assert "s3://test-bucket-uuid/data/" in cmd

    @patch("lambda_cloud_toolkit.storage.subprocess.run")
    def test_ls_root(self, mock_run, storage):
        mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
        storage.ls()
        cmd = mock_run.call_args[0][0]
        assert "s3://test-bucket-uuid/" in cmd


class TestFromConfig:
    def test_creates_from_config_dict(self):
        env = {"LAMBDA_ACCESS_KEY_ID": "k", "LAMBDA_SECRET_ACCESS_KEY": "s"}
        with patch.dict(os.environ, env, clear=False):
            s = LambdaStorage.from_config({
                "bucket_name": "cfg-bucket",
                "endpoint_url": "https://cfg.endpoint",
                "region": "us-west-2",
            })
            assert s.bucket_name == "cfg-bucket"
            assert s.endpoint_url == "https://cfg.endpoint"
            assert s.region == "us-west-2"
