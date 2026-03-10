"""Tests for lambda_cloud_toolkit.utils."""

import os
from unittest.mock import patch, mock_open

import pytest

from lambda_cloud_toolkit.utils import find_env_file, load_env_file


class TestFindEnvFile:
    @patch("lambda_cloud_toolkit.utils.os.path.isfile", return_value=True)
    def test_finds_dot_sync_env(self, mock_isfile):
        assert find_env_file() == ".sync.env"

    @patch("lambda_cloud_toolkit.utils._glob.glob", return_value=["myname.sync.env"])
    @patch("lambda_cloud_toolkit.utils.os.path.isfile", return_value=False)
    def test_finds_single_named_env(self, mock_isfile, mock_glob):
        assert find_env_file() == "myname.sync.env"

    @patch("lambda_cloud_toolkit.utils._glob.glob", return_value=["a.sync.env", "b.sync.env"])
    @patch("lambda_cloud_toolkit.utils.os.path.isfile", return_value=False)
    def test_returns_none_for_multiple(self, mock_isfile, mock_glob):
        assert find_env_file() is None

    @patch("lambda_cloud_toolkit.utils._glob.glob", return_value=[])
    @patch("lambda_cloud_toolkit.utils.os.path.isfile", return_value=False)
    def test_returns_none_when_missing(self, mock_isfile, mock_glob):
        assert find_env_file() is None


class TestLoadEnvFile:
    def test_loads_export_lines(self, tmp_path):
        env_file = tmp_path / ".sync.env"
        env_file.write_text("export MY_TEST_VAR_123=hello\n")
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MY_TEST_VAR_123", None)
            load_env_file(str(env_file))
            assert os.environ["MY_TEST_VAR_123"] == "hello"
            del os.environ["MY_TEST_VAR_123"]

    def test_loads_lines_without_export(self, tmp_path):
        env_file = tmp_path / ".sync.env"
        env_file.write_text("MY_TEST_VAR_456=world\n")
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MY_TEST_VAR_456", None)
            load_env_file(str(env_file))
            assert os.environ["MY_TEST_VAR_456"] == "world"
            del os.environ["MY_TEST_VAR_456"]

    def test_does_not_override_existing(self, tmp_path):
        env_file = tmp_path / ".sync.env"
        env_file.write_text("export MY_TEST_VAR_789=new\n")
        with patch.dict(os.environ, {"MY_TEST_VAR_789": "existing"}, clear=False):
            load_env_file(str(env_file))
            assert os.environ["MY_TEST_VAR_789"] == "existing"

    def test_strips_quotes(self, tmp_path):
        env_file = tmp_path / ".sync.env"
        env_file.write_text('export MY_TEST_VAR_ABC="quoted value"\n')
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MY_TEST_VAR_ABC", None)
            load_env_file(str(env_file))
            assert os.environ["MY_TEST_VAR_ABC"] == "quoted value"
            del os.environ["MY_TEST_VAR_ABC"]

    def test_skips_comments_and_blanks(self, tmp_path):
        env_file = tmp_path / ".sync.env"
        env_file.write_text("# comment\n\nexport MY_TEST_VAR_DEF=yes\n")
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MY_TEST_VAR_DEF", None)
            load_env_file(str(env_file))
            assert os.environ["MY_TEST_VAR_DEF"] == "yes"
            del os.environ["MY_TEST_VAR_DEF"]
