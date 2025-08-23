import os
import logging
from io import BytesIO
from minio import Minio
from minio.error import S3Error
import dotenv
dotenv.load_dotenv()

logger = logging.getLogger(__name__)

def init_minio_client():
    """Initialize and configure the MinIO client."""
    endpoint = os.getenv("MINIO_ENDPOINT")
    access_key = os.getenv("MINIO_ACCESS_KEY")
    secret_key = os.getenv("MINIO_SECRET_KEY")
    bucket_name = os.getenv("MINIO_BUCKET")

    required_vars = ["MINIO_ENDPOINT", "MINIO_ACCESS_KEY", "MINIO_SECRET_KEY", "MINIO_BUCKET"]
    for var in required_vars:
        if not os.getenv(var):
            raise EnvironmentError(f"Missing required environment variable: {var}")

    # Use localhost:9000 for local runs, minio:9000 in Docker
    if os.getenv("RUNNING_IN_DOCKER") is None:
        endpoint = "localhost:9000"  # Override for local execution
        logger.info("Running locally, using MinIO endpoint: localhost:9000")
    else:
        logger.info(f"Running in Docker, using MinIO endpoint: {endpoint}")

    client = Minio(
        endpoint,
        access_key=access_key,
        secret_key=secret_key,
        secure=False
    )

    try:
        if not client.bucket_exists(bucket_name):
            client.make_bucket(bucket_name)
            logger.info(f"Created MinIO bucket: {bucket_name}")
        else:
            logger.info(f"MinIO bucket already exists: {bucket_name}")
    except S3Error as e:
        logger.error(f"Failed to initialize MinIO bucket: {e}")
        raise

    return client, bucket_name

minio_client, bucket_name = init_minio_client()

async def upload_to_minio(file_data: bytes, file_name: str) -> bool:
    """Upload bytes directly to MinIO and return success status."""
    try:
        minio_client.put_object(
            bucket_name=bucket_name,
            object_name=file_name,
            data=BytesIO(file_data),
            length=len(file_data),
            part_size=5 * 1024 * 1024
        )
        logger.info(f"Uploaded {file_name} to MinIO bucket {bucket_name}")
        return True
    except S3Error as e:
        logger.error(f"Failed to upload {file_name} to MinIO: {e}")
        return False