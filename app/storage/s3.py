from __future__ import annotations
import logging
import tempfile
from pathlib import Path
import aioboto3
from app.config import get_settings

logger = logging.getLogger("ragdms.s3")

async def download_to_tempfile(s3_key: str, *, suffix: str = "") -> Path:
    """Download an S3 object to a local temp file. Caller must delete it when done."""
    settings = get_settings()
    session = aioboto3.Session(
        aws_access_key_id=settings.aws_access_key_id or None,
        aws_secret_access_key=settings.aws_secret_access_key or None,
        region_name=settings.aws_region,
    )
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp_path = Path(tmp.name)
    tmp.close()
    try:
        async with session.client("s3") as s3:
            await s3.download_file(settings.s3_bucket, s3_key, str(tmp_path))
        logger.info("Downloaded s3://%s/%s -> %s", settings.s3_bucket, s3_key, tmp_path)
        return tmp_path
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
