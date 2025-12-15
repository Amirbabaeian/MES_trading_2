"""
Azure Blob Storage provider implementation.

Features:
- Integration with Azure Blob Storage containers
- Configurable access tiers (hot, cool, archive)
- Upload and download with streaming
"""

import logging
from pathlib import Path
from typing import Optional, List, Callable
from datetime import datetime, timezone
import hashlib

from ..cloud_sync import CloudStorage, FileMetadata

logger = logging.getLogger(__name__)

try:
    from azure.storage.blob import BlobServiceClient, BlobSasPermissions, generate_blob_sas
    from azure.core.exceptions import ResourceNotFoundError, AzureError
    HAS_AZURE = True
except ImportError:
    HAS_AZURE = False
    logger.warning("azure-storage-blob not installed. Azure provider will not be available.")


class AzureCloudStorage(CloudStorage):
    """Azure Blob Storage implementation of CloudStorage."""
    
    DEFAULT_CHUNK_SIZE = 5 * 1024 * 1024  # 5 MB
    
    def __init__(
        self,
        container: str,
        connection_string: Optional[str] = None,
        account_name: Optional[str] = None,
        account_key: Optional[str] = None,
        access_tier: str = "hot",
    ):
        """
        Initialize Azure Blob Storage provider.
        
        Args:
            container: Container name
            connection_string: Connection string (preferred)
            account_name: Storage account name
            account_key: Storage account key
            access_tier: Access tier (hot, cool, archive)
        
        Raises:
            ImportError: If azure-storage-blob is not installed
            ValueError: If neither connection_string nor account credentials provided
        """
        if not HAS_AZURE:
            raise ImportError(
                "azure-storage-blob is required for Azure support. "
                "Install with: pip install azure-storage-blob"
            )
        
        self.container_name = container
        self.connection_string = connection_string
        self.account_name = account_name
        self.account_key = account_key
        self.access_tier = access_tier
        self._client = None
        self._container_client = None
        
        # Validate access tier
        valid_tiers = {"hot", "cool", "archive"}
        if access_tier not in valid_tiers:
            raise ValueError(f"Invalid access tier: {access_tier}")
        
        if not connection_string and not (account_name and account_key):
            raise ValueError("Provide either connection_string or account_name/account_key")
    
    def connect(self) -> None:
        """Establish connection to Azure Blob Storage."""
        try:
            if self.connection_string:
                self._client = BlobServiceClient.from_connection_string(
                    self.connection_string
                )
            else:
                account_url = f"https://{self.account_name}.blob.core.windows.net"
                self._client = BlobServiceClient(
                    account_url=account_url,
                    credential=self.account_key,
                )
            
            # Verify container exists
            self._container_client = self._client.get_container_client(self.container_name)
            if not self._container_client.exists():
                raise ValueError(f"Container {self.container_name} does not exist")
            
            logger.info(f"Connected to Azure Blob Storage container: {self.container_name}")
        
        except Exception as e:
            raise RuntimeError(f"Failed to connect to Azure Blob Storage: {e}")
    
    def disconnect(self) -> None:
        """Close connection to Azure Blob Storage."""
        if self._client:
            logger.info("Disconnected from Azure Blob Storage")
    
    def upload_file(
        self,
        local_path: Path,
        cloud_path: str,
        multipart_threshold: int = 100 * 1024 * 1024,
        progress_callback: Optional[Callable[[int], None]] = None,
    ) -> str:
        """Upload a file to Azure Blob Storage."""
        if not self._container_client:
            raise RuntimeError("Not connected to Azure. Call connect() first.")
        
        local_path = Path(local_path)
        file_size = local_path.stat().st_size
        
        logger.info(f"Uploading {local_path} to {self.container_name}/{cloud_path}")
        
        try:
            blob_client = self._container_client.get_blob_client(cloud_path)
            
            hash_md5 = hashlib.md5()
            with open(local_path, "rb") as f:
                data = f.read()
                hash_md5.update(data)
                blob_client.upload_blob(data, overwrite=True)
            
            if progress_callback:
                progress_callback(file_size)
            
            checksum = hash_md5.hexdigest()
            logger.info(f"Uploaded {cloud_path} (MD5: {checksum})")
            return checksum
        
        except AzureError as e:
            logger.error(f"Azure upload failed: {e}")
            raise
    
    def download_file(
        self,
        cloud_path: str,
        local_path: Path,
        progress_callback: Optional[Callable[[int], None]] = None,
    ) -> str:
        """Download a file from Azure Blob Storage."""
        if not self._container_client:
            raise RuntimeError("Not connected to Azure. Call connect() first.")
        
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Downloading {self.container_name}/{cloud_path} to {local_path}")
        
        try:
            blob_client = self._container_client.get_blob_client(cloud_path)
            
            hash_md5 = hashlib.md5()
            with open(local_path, "wb") as f:
                download_stream = blob_client.download_blob()
                for chunk in download_stream.chunks():
                    f.write(chunk)
                    hash_md5.update(chunk)
                    if progress_callback:
                        progress_callback(len(chunk))
            
            checksum = hash_md5.hexdigest()
            logger.info(f"Downloaded {cloud_path} (MD5: {checksum})")
            return checksum
        
        except AzureError as e:
            logger.error(f"Azure download failed: {e}")
            raise
    
    def delete_file(self, cloud_path: str) -> None:
        """Delete a file from Azure Blob Storage."""
        if not self._container_client:
            raise RuntimeError("Not connected to Azure. Call connect() first.")
        
        try:
            blob_client = self._container_client.get_blob_client(cloud_path)
            blob_client.delete_blob()
            logger.info(f"Deleted {self.container_name}/{cloud_path}")
        except AzureError as e:
            logger.error(f"Failed to delete {cloud_path}: {e}")
            raise
    
    def list_files(
        self,
        prefix: str = "",
        recursive: bool = True,
    ) -> List[FileMetadata]:
        """List files in Azure Blob Storage."""
        if not self._container_client:
            raise RuntimeError("Not connected to Azure. Call connect() first.")
        
        logger.info(f"Listing files in {self.container_name}/{prefix}")
        
        try:
            files = []
            for blob in self._container_client.list_blobs(name_starts_with=prefix):
                # Skip directories
                if blob.name.endswith('/'):
                    continue
                
                file_metadata = FileMetadata(
                    path=blob.name,
                    size=blob.size,
                    modified_time=blob.last_modified,
                    checksum=None,  # Azure doesn't reliably expose checksums
                    storage_class=None,
                )
                files.append(file_metadata)
            
            logger.info(f"Found {len(files)} files in {self.container_name}/{prefix}")
            return files
        
        except AzureError as e:
            logger.error(f"Failed to list Azure blobs: {e}")
            raise
    
    def file_exists(self, cloud_path: str) -> bool:
        """Check if a file exists in Azure Blob Storage."""
        if not self._container_client:
            raise RuntimeError("Not connected to Azure. Call connect() first.")
        
        try:
            blob_client = self._container_client.get_blob_client(cloud_path)
            return blob_client.exists()
        except AzureError:
            return False
    
    def get_file_metadata(self, cloud_path: str) -> Optional[FileMetadata]:
        """Get metadata for a file in Azure Blob Storage."""
        if not self._container_client:
            raise RuntimeError("Not connected to Azure. Call connect() first.")
        
        try:
            blob_client = self._container_client.get_blob_client(cloud_path)
            if not blob_client.exists():
                return None
            
            properties = blob_client.get_blob_properties()
            
            return FileMetadata(
                path=cloud_path,
                size=properties.size,
                modified_time=properties.last_modified,
                checksum=None,
                storage_class=properties.access_tier,
            )
        except AzureError:
            return None
