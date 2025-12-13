"""
Google Cloud Storage (GCS) provider implementation.

Features:
- Integration with GCS buckets
- Configurable storage classes (STANDARD, NEARLINE, COLDLINE, ARCHIVE)
- Streaming uploads and downloads
"""

import logging
from pathlib import Path
from typing import Optional, List, Callable
from datetime import datetime, timezone
import hashlib

from ..cloud_sync import CloudStorage, FileMetadata

logger = logging.getLogger(__name__)

try:
    from google.cloud import storage
    from google.api_core.exceptions import NotFound, GoogleAPICallError
    HAS_GCS = True
except ImportError:
    HAS_GCS = False
    logger.warning("google-cloud-storage not installed. GCS provider will not be available.")


class GCSCloudStorage(CloudStorage):
    """Google Cloud Storage implementation of CloudStorage."""
    
    DEFAULT_CHUNK_SIZE = 5 * 1024 * 1024  # 5 MB
    
    def __init__(
        self,
        bucket: str,
        project_id: Optional[str] = None,
        credentials_path: Optional[str] = None,
        storage_class: str = "STANDARD",
    ):
        """
        Initialize GCS cloud storage provider.
        
        Args:
            bucket: GCS bucket name
            project_id: Google Cloud project ID
            credentials_path: Path to service account JSON
            storage_class: Storage class (STANDARD, NEARLINE, COLDLINE, ARCHIVE)
        
        Raises:
            ImportError: If google-cloud-storage is not installed
        """
        if not HAS_GCS:
            raise ImportError(
                "google-cloud-storage is required for GCS support. "
                "Install with: pip install google-cloud-storage"
            )
        
        self.bucket_name = bucket
        self.project_id = project_id
        self.credentials_path = credentials_path
        self.storage_class = storage_class
        self._client = None
        self._bucket = None
        
        # Validate storage class
        valid_classes = {"STANDARD", "NEARLINE", "COLDLINE", "ARCHIVE"}
        if storage_class not in valid_classes:
            raise ValueError(f"Invalid storage class: {storage_class}")
    
    def connect(self) -> None:
        """Establish connection to GCS."""
        try:
            if self.credentials_path:
                self._client = storage.Client.from_service_account_json(
                    self.credentials_path,
                    project=self.project_id,
                )
            else:
                self._client = storage.Client(project=self.project_id)
            
            # Verify bucket exists
            self._bucket = self._client.bucket(self.bucket_name)
            if not self._bucket.exists():
                raise ValueError(f"Bucket {self.bucket_name} does not exist")
            
            logger.info(f"Connected to GCS bucket: {self.bucket_name}")
        
        except Exception as e:
            raise RuntimeError(f"Failed to connect to GCS: {e}")
    
    def disconnect(self) -> None:
        """Close connection to GCS."""
        if self._client:
            logger.info("Disconnected from GCS")
    
    def upload_file(
        self,
        local_path: Path,
        cloud_path: str,
        multipart_threshold: int = 100 * 1024 * 1024,
        progress_callback: Optional[Callable[[int], None]] = None,
    ) -> str:
        """Upload a file to GCS."""
        if not self._bucket:
            raise RuntimeError("Not connected to GCS. Call connect() first.")
        
        local_path = Path(local_path)
        file_size = local_path.stat().st_size
        
        logger.info(f"Uploading {local_path} to gs://{self.bucket_name}/{cloud_path}")
        
        try:
            blob = self._bucket.blob(cloud_path)
            
            with open(local_path, "rb") as f:
                blob.upload_from_file(f)
            
            if progress_callback:
                progress_callback(file_size)
            
            logger.info(f"Uploaded {cloud_path} (MD5: {blob.md5_hash})")
            return blob.md5_hash or "unknown"
        
        except GoogleAPICallError as e:
            logger.error(f"GCS upload failed: {e}")
            raise
    
    def download_file(
        self,
        cloud_path: str,
        local_path: Path,
        progress_callback: Optional[Callable[[int], None]] = None,
    ) -> str:
        """Download a file from GCS."""
        if not self._bucket:
            raise RuntimeError("Not connected to GCS. Call connect() first.")
        
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Downloading gs://{self.bucket_name}/{cloud_path} to {local_path}")
        
        try:
            blob = self._bucket.blob(cloud_path)
            
            hash_md5 = hashlib.md5()
            with open(local_path, "wb") as f:
                self._client.download_blob_to_file(blob, f)
                f.seek(0)
                while True:
                    chunk = f.read(self.DEFAULT_CHUNK_SIZE)
                    if not chunk:
                        break
                    hash_md5.update(chunk)
                    if progress_callback:
                        progress_callback(len(chunk))
            
            checksum = hash_md5.hexdigest()
            logger.info(f"Downloaded {cloud_path} (MD5: {checksum})")
            return checksum
        
        except GoogleAPICallError as e:
            logger.error(f"GCS download failed: {e}")
            raise
    
    def delete_file(self, cloud_path: str) -> None:
        """Delete a file from GCS."""
        if not self._bucket:
            raise RuntimeError("Not connected to GCS. Call connect() first.")
        
        try:
            blob = self._bucket.blob(cloud_path)
            blob.delete()
            logger.info(f"Deleted gs://{self.bucket_name}/{cloud_path}")
        except GoogleAPICallError as e:
            logger.error(f"Failed to delete {cloud_path}: {e}")
            raise
    
    def list_files(
        self,
        prefix: str = "",
        recursive: bool = True,
    ) -> List[FileMetadata]:
        """List files in GCS."""
        if not self._bucket:
            raise RuntimeError("Not connected to GCS. Call connect() first.")
        
        logger.info(f"Listing files in gs://{self.bucket_name}/{prefix}")
        
        try:
            files = []
            for blob in self._bucket.list_blobs(prefix=prefix):
                # Skip directories
                if blob.name.endswith('/'):
                    continue
                
                file_metadata = FileMetadata(
                    path=blob.name,
                    size=blob.size,
                    modified_time=blob.updated,
                    checksum=blob.md5_hash,
                    storage_class=blob.storage_class,
                )
                files.append(file_metadata)
            
            logger.info(f"Found {len(files)} files in gs://{self.bucket_name}/{prefix}")
            return files
        
        except GoogleAPICallError as e:
            logger.error(f"Failed to list GCS objects: {e}")
            raise
    
    def file_exists(self, cloud_path: str) -> bool:
        """Check if a file exists in GCS."""
        if not self._bucket:
            raise RuntimeError("Not connected to GCS. Call connect() first.")
        
        try:
            blob = self._bucket.blob(cloud_path)
            return blob.exists()
        except GoogleAPICallError:
            return False
    
    def get_file_metadata(self, cloud_path: str) -> Optional[FileMetadata]:
        """Get metadata for a file in GCS."""
        if not self._bucket:
            raise RuntimeError("Not connected to GCS. Call connect() first.")
        
        try:
            blob = self._bucket.blob(cloud_path)
            if not blob.exists():
                return None
            
            return FileMetadata(
                path=blob.name,
                size=blob.size,
                modified_time=blob.updated,
                checksum=blob.md5_hash,
                storage_class=blob.storage_class,
            )
        except GoogleAPICallError:
            return None
