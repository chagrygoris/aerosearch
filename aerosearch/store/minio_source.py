"""MinIO-backed Downloader for the batch indexer.

The bot stores `s3_url` as ``{endpoint}/{bucket}/{object_name}``; we pull the
object bytes back by name. Kept out of the batch module so aerosearch has no hard
dependency on the minio client unless this is actually used.
"""
from __future__ import annotations

import os


class MinioDownloader:
    def __init__(
        self,
        endpoint: str | None = None,
        access_key: str | None = None,
        secret_key: str | None = None,
        bucket: str | None = None,
        secure: bool = False,
    ):
        self.endpoint = endpoint or os.getenv("MINIO_ENDPOINT", "localhost:9000")
        self.access_key = access_key or os.getenv("MINIO_ACCESS_KEY")
        self.secret_key = secret_key or os.getenv("MINIO_SECRET_KEY")
        self.bucket = bucket or os.getenv("MINIO_BUCKET")
        self.secure = secure
        self._client = None

    def _get_client(self):
        if self._client is None:
            from minio import Minio

            self._client = Minio(
                self.endpoint,
                access_key=self.access_key,
                secret_key=self.secret_key,
                secure=self.secure,
            )
        return self._client

    def _object_name(self, s3_url: str) -> str:
        # Strip a leading "{bucket}/" if the stored url includes it.
        marker = f"{self.bucket}/"
        if marker in s3_url:
            return s3_url.split(marker, 1)[1]
        return s3_url.rsplit("/", 1)[-1]

    def download(self, s3_url: str) -> bytes:
        client = self._get_client()
        obj = self._object_name(s3_url)
        response = client.get_object(self.bucket, obj)
        try:
            return response.read()
        finally:
            response.close()
            response.release_conn()
