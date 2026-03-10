"""S3-compatible storage sync for Lambda Cloud Filesystem.

Wraps the `aws` CLI via subprocess — Lambda Filesystem exposes an
S3-compatible endpoint, so we reuse `awscli` rather than adding boto3.
Requires `aws` to be available on PATH.
"""

import logging
import os
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


class LambdaStorage:
    """S3-compatible storage operations against Lambda Cloud Filesystem.

    Args:
        bucket_name: S3 bucket name (UUID from Lambda Filesystem).
        endpoint_url: S3-compatible endpoint (default: Lambda US-East-2).
        region: AWS region for the endpoint.
        access_key_id: Lambda S3 access key (or reads LAMBDA_ACCESS_KEY_ID env var).
        secret_access_key: Lambda S3 secret key (or reads LAMBDA_SECRET_ACCESS_KEY env var).
        syncignore: Path to .syncignore file for exclusion patterns.
    """

    DEFAULT_ENDPOINT = "https://files.us-east-2.lambda.ai"
    DEFAULT_REGION = "us-east-2"

    def __init__(
        self,
        bucket_name: str | None = None,
        endpoint_url: str | None = None,
        region: str | None = None,
        access_key_id: str | None = None,
        secret_access_key: str | None = None,
        syncignore: str | None = None,
    ):
        self.bucket_name = bucket_name or os.environ.get("BUCKET_NAME", "")
        self.endpoint_url = endpoint_url or os.environ.get("LAMBDA_ENDPOINT_URL", self.DEFAULT_ENDPOINT)
        self.region = region or os.environ.get("LAMBDA_REGION", self.DEFAULT_REGION)
        self._access_key_id = access_key_id or os.environ.get("LAMBDA_ACCESS_KEY_ID", "")
        self._secret_access_key = secret_access_key or os.environ.get("LAMBDA_SECRET_ACCESS_KEY", "")
        self._syncignore = syncignore

        if not self.bucket_name:
            raise ValueError(
                "bucket_name is required — pass it directly or set BUCKET_NAME env var"
            )
        if not self._access_key_id or not self._secret_access_key:
            raise ValueError(
                "S3 credentials required — pass access_key_id/secret_access_key "
                "or set LAMBDA_ACCESS_KEY_ID/LAMBDA_SECRET_ACCESS_KEY env vars"
            )

    def _env(self) -> dict[str, str]:
        """Build environment variables for aws CLI subprocess."""
        env = os.environ.copy()
        env.update({
            "AWS_ACCESS_KEY_ID": self._access_key_id,
            "AWS_SECRET_ACCESS_KEY": self._secret_access_key,
            "AWS_DEFAULT_REGION": self.region,
            # Lambda S3 adapter requires these to avoid NotImplemented checksum errors
            "AWS_REQUEST_CHECKSUM_CALCULATION": "when_required",
            "AWS_RESPONSE_CHECKSUM_VALIDATION": "when_required",
        })
        return env

    def _exclude_flags(self) -> list[str]:
        """Parse .syncignore and return --exclude flags for aws s3 sync."""
        if not self._syncignore or not os.path.isfile(self._syncignore):
            return []
        flags = []
        with open(self._syncignore) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                flags.extend(["--exclude", line])
        return flags

    def _bucket_uri(self, subpath: str | None = None) -> str:
        """Build s3://bucket/subpath URI."""
        base = f"s3://{self.bucket_name}"
        if subpath:
            return f"{base}/{subpath.strip('/')}/"
        return f"{base}/"

    def upload(self, local_dir: str, subpath: str | None = None) -> subprocess.CompletedProcess:
        """Sync local directory to S3 bucket.

        Args:
            local_dir: Local directory path to upload from.
            subpath: Optional subdirectory within the bucket (e.g., "phase0/data/results").

        Returns:
            CompletedProcess from the aws s3 sync command.

        Raises:
            FileNotFoundError: If local_dir doesn't exist.
            subprocess.CalledProcessError: If aws CLI fails.
        """
        local_path = Path(local_dir)
        if not local_path.is_dir():
            raise FileNotFoundError(f"Local directory not found: {local_dir}")

        bucket_uri = self._bucket_uri(subpath)
        cmd = [
            "aws", "s3", "sync",
            str(local_path) + "/",
            bucket_uri,
            "--endpoint-url", self.endpoint_url,
            *self._exclude_flags(),
        ]
        logger.info("Uploading %s → %s", local_dir, bucket_uri)
        return subprocess.run(cmd, env=self._env(), check=True, capture_output=True, text=True)

    def download(self, local_dir: str, subpath: str | None = None) -> subprocess.CompletedProcess:
        """Sync S3 bucket to local directory.

        Args:
            local_dir: Local directory path to download into.
            subpath: Optional subdirectory within the bucket.

        Returns:
            CompletedProcess from the aws s3 sync command.

        Raises:
            subprocess.CalledProcessError: If aws CLI fails.
        """
        local_path = Path(local_dir)
        local_path.mkdir(parents=True, exist_ok=True)

        bucket_uri = self._bucket_uri(subpath)
        cmd = [
            "aws", "s3", "sync",
            bucket_uri,
            str(local_path) + "/",
            "--endpoint-url", self.endpoint_url,
            *self._exclude_flags(),
        ]
        logger.info("Downloading %s → %s", bucket_uri, local_dir)
        return subprocess.run(cmd, env=self._env(), check=True, capture_output=True, text=True)

    def ls(self, path: str | None = None) -> subprocess.CompletedProcess:
        """List contents of the S3 bucket.

        Args:
            path: Optional path within the bucket to list.

        Returns:
            CompletedProcess with stdout containing the listing.
        """
        target = self._bucket_uri(path)
        cmd = [
            "aws", "s3", "ls",
            target,
            "--endpoint-url", self.endpoint_url,
        ]
        logger.info("Listing %s", target)
        return subprocess.run(cmd, env=self._env(), check=True, capture_output=True, text=True)

    @classmethod
    def from_config(cls, config: dict, syncignore: str | None = None) -> "LambdaStorage":
        """Create LambdaStorage from a config dict (storage section of lambda-cloud.yaml).

        Args:
            config: Dict with keys: bucket_name, endpoint_url, region.
            syncignore: Path to .syncignore file.

        Returns:
            Configured LambdaStorage instance.
        """
        return cls(
            bucket_name=config.get("bucket_name"),
            endpoint_url=config.get("endpoint_url"),
            region=config.get("region"),
            syncignore=syncignore or config.get("syncignore"),
        )
