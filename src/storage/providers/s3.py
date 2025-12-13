"""
AWS S3 cloud storage provider implementation.

Features:
- Multipart upload for large files
- Streaming downloads
- Configurable storage classes (STANDARD, STANDARD_IA, GLACIER)
- Retry logic for transient failures
"""

import logging
from pathlib import Path
from typing import Optional, List, Callable
from datetime import datetime, timezone
import hashlib

from ..cloud_sync import CloudStorage, FileMetadata

logger = logging.getLogger(__name__)

try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
    from botocore.config import Config
    HAS_BOTO3 = True
except ImportError:
    HAS_BOTO3 = False
    logger.warning("boto3 not installed. S3 provider will not be available.")


class S3CloudStorage(CloudStorage):
    """AWS S3 implementation of CloudStorage."""
    
    # Multipart upload chunk size (5 MB minimum for S3)
    DEFAULT_CHUNK_SIZE = 5 * 1024 * 1024  # 5 MB
    
    def __init__(
        self,
        bucket: str,
        region: str = "us-east-1",
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        storage_class: str = "STANDARD",
        create_bucket_if_missing: bool = False,
    ):
        """
        Initialize S3 cloud storage provider.
        
        Args:
            bucket: S3 bucket name
            region: AWS region
            aws_access_key_id: AWS access key (uses env if not provided)
            aws_secret_access_key: AWS secret key (uses env if not provided)
            storage_class: Storage class (STANDARD, STANDARD_IA, GLACIER)
            create_bucket_if_missing: Whether to create bucket if it doesn't exist
        
        Raises:
            ImportError: If boto3 is not installed
            ValueError: If bucket name is invalid
        """
        if not HAS_BOTO3:
            raise ImportError("boto3 is required for S3 support. Install with: pip install boto3")
        
        self.bucket = bucket
        self.region = region
        self.storage_class = storage_class
        self.create_bucket_if_missing = create_bucket_if_missing
        self._client = None
        self._resource = None
        self._session = None
        
        # Validate storage class
        valid_classes = {"STANDARD", "STANDARD_IA", "GLACIER", "DEEP_ARCHIVE"}
        if storage_class not in valid_classes:
            raise ValueError(f"Invalid storage class: {storage_class}")
        
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
    
    def connect(self) -> None:
        """Establish connection to S3."""
        try:
            # Create session
            if self.aws_access_key_id and self.aws_secret_access_key:
                self._session = boto3.Session(
                    aws_access_key_id=self.aws_access_key_id,
                    aws_secret_access_key=self.aws_secret_access_key,
                    region_name=self.region,
                )
            else:
                self._session = boto3.Session(region_name=self.region)
            
            # Create client with retry config
            config = Config(
                retries={'max_attempts': 3, 'mode': 'adaptive'},
                max_pool_connections=10,
            )
            self._client = self._session.client("s3", config=config)
            self._resource = self._session.resource("s3")
            
            # Verify bucket exists or create it
            try:
                self._client.head_bucket(Bucket=self.bucket)
                logger.info(f"Connected to S3 bucket: {self.bucket}")
            except ClientError as e:
                if e.response['Error']['Code'] == '404':
                    if self.create_bucket_if_missing:
                        logger.info(f"Creating S3 bucket: {self.bucket}")
                        if self.region == "us-east-1":
                            self._client.create_bucket(Bucket=self.bucket)
                        else:
                            self._client.create_bucket(
                                Bucket=self.bucket,
                                CreateBucketConfiguration={'LocationConstraint': self.region}
                            )
                    else:
                        raise ValueError(f"Bucket {self.bucket} does not exist")
                else:
                    raise
        
        except NoCredentialsError:
            raise ValueError("AWS credentials not found. Configure via environment or parameters.")
        except Exception as e:
            raise RuntimeError(f"Failed to connect to S3: {e}")
    
    def disconnect(self) -> None:
        """Close connection to S3."""
        if self._client:
            self._client.close()
            logger.info("Disconnected from S3")
    
    def upload_file(
        self,
        local_path: Path,
        cloud_path: str,
        multipart_threshold: int = 100 * 1024 * 1024,
        progress_callback: Optional[Callable[[int], None]] = None,
    ) -> str:
        """
        Upload a file to S3.
        
        Args:
            local_path: Local file path
            cloud_path: S3 destination path (key)
            multipart_threshold: Use multipart for files larger than this
            progress_callback: Optional callback for progress
        
        Returns:
            ETag of uploaded file
        """
        if not self._client:
            raise RuntimeError("Not connected to S3. Call connect() first.")
        
        local_path = Path(local_path)
        file_size = local_path.stat().st_size
        
        logger.info(f"Uploading {local_path} to s3://{self.bucket}/{cloud_path} ({file_size} bytes)")
        
        try:
            if file_size > multipart_threshold:
                # Use multipart upload
                return self._multipart_upload(
                    local_path, cloud_path, file_size, progress_callback
                )
            else:
                # Use simple upload
                return self._simple_upload(local_path, cloud_path, progress_callback)
        
        except ClientError as e:
            logger.error(f"S3 upload failed: {e}")
            raise
    
    def _simple_upload(
        self,
        local_path: Path,
        cloud_path: str,
        progress_callback: Optional[Callable[[int], None]] = None,
    ) -> str:
        """Simple upload for small files."""
        with open(local_path, "rb") as f:
            data = f.read()
            
            response = self._client.put_object(
                Bucket=self.bucket,
                Key=cloud_path,
                Body=data,
                StorageClass=self.storage_class,
            )
            
            if progress_callback:
                progress_callback(len(data))
            
            etag = response['ETag'].strip('"')
            logger.debug(f"Uploaded {local_path} (ETag: {etag})")
            return etag
    
    def _multipart_upload(
        self,
        local_path: Path,
        cloud_path: str,
        file_size: int,
        progress_callback: Optional[Callable[[int], None]] = None,
    ) -> str:
        """Multipart upload for large files."""
        # Initiate multipart upload
        response = self._client.create_multipart_upload(
            Bucket=self.bucket,
            Key=cloud_path,
            StorageClass=self.storage_class,
        )
        upload_id = response['UploadId']
        
        logger.debug(f"Started multipart upload {upload_id} for {local_path}")
        
        parts = []
        try:
            with open(local_path, "rb") as f:
                part_num = 1
                
                while True:
                    chunk = f.read(self.DEFAULT_CHUNK_SIZE)
                    if not chunk:
                        break
                    
                    logger.debug(f"Uploading part {part_num} ({len(chunk)} bytes)")
                    
                    part_response = self._client.upload_part(
                        Bucket=self.bucket,
                        Key=cloud_path,
                        PartNumber=part_num,
                        UploadId=upload_id,
                        Body=chunk,
                    )
                    
                    parts.append({
                        'ETag': part_response['ETag'].strip('"'),
                        'PartNumber': part_num,
                    })
                    
                    if progress_callback:
                        progress_callback(len(chunk))
                    
                    part_num += 1
            
            # Complete multipart upload
            response = self._client.complete_multipart_upload(
                Bucket=self.bucket,
                Key=cloud_path,
                UploadId=upload_id,
                MultipartUpload={'Parts': parts},
            )
            
            etag = response['ETag'].strip('"')
            logger.info(f"Completed multipart upload {upload_id} (ETag: {etag})")
            return etag
        
        except Exception as e:
            logger.error(f"Multipart upload failed, aborting {upload_id}")
            self._client.abort_multipart_upload(
                Bucket=self.bucket,
                Key=cloud_path,
                UploadId=upload_id,
            )
            raise
    
    def download_file(
        self,
        cloud_path: str,
        local_path: Path,
        progress_callback: Optional[Callable[[int], None]] = None,
    ) -> str:
        """
        Download a file from S3.
        
        Args:
            cloud_path: S3 source path (key)
            local_path: Local destination path
            progress_callback: Optional callback for progress
        
        Returns:
            MD5 checksum of downloaded file
        """
        if not self._client:
            raise RuntimeError("Not connected to S3. Call connect() first.")
        
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Downloading s3://{self.bucket}/{cloud_path} to {local_path}")
        
        try:
            # Get file metadata
            response = self._client.head_object(Bucket=self.bucket, Key=cloud_path)
            file_size = response['ContentLength']
            
            # Download with streaming
            hash_md5 = hashlib.md5()
            bytes_downloaded = 0
            
            with self._client.get_object(Bucket=self.bucket, Key=cloud_path)['Body'] as f:
                with open(local_path, "wb") as local_file:
                    while True:
                        chunk = f.read(self.DEFAULT_CHUNK_SIZE)
                        if not chunk:
                            break
                        
                        local_file.write(chunk)
                        hash_md5.update(chunk)
                        bytes_downloaded += len(chunk)
                        
                        if progress_callback:
                            progress_callback(len(chunk))
            
            checksum = hash_md5.hexdigest()
            logger.info(f"Downloaded {cloud_path} ({bytes_downloaded} bytes, MD5: {checksum})")
            return checksum
        
        except ClientError as e:
            logger.error(f"S3 download failed: {e}")
            raise
    
    def delete_file(self, cloud_path: str) -> None:
        """Delete a file from S3."""
        if not self._client:
            raise RuntimeError("Not connected to S3. Call connect() first.")
        
        try:
            self._client.delete_object(Bucket=self.bucket, Key=cloud_path)
            logger.info(f"Deleted s3://{self.bucket}/{cloud_path}")
        except ClientError as e:
            logger.error(f"Failed to delete {cloud_path}: {e}")
            raise
    
    def list_files(
        self,
        prefix: str = "",
        recursive: bool = True,
    ) -> List[FileMetadata]:
        """
        List files in S3.
        
        Args:
            prefix: Optional prefix to filter files
            recursive: Whether to list recursively (always true for S3)
        
        Returns:
            List of FileMetadata objects
        """
        if not self._client:
            raise RuntimeError("Not connected to S3. Call connect() first.")
        
        logger.info(f"Listing files in s3://{self.bucket}/{prefix}")
        
        files = []
        paginator = self._client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=self.bucket, Prefix=prefix)
        
        try:
            for page in pages:
                if 'Contents' not in page:
                    continue
                
                for obj in page['Contents']:
                    # Skip directories (keys ending with /)
                    if obj['Key'].endswith('/'):
                        continue
                    
                    file_metadata = FileMetadata(
                        path=obj['Key'],
                        size=obj['Size'],
                        modified_time=obj['LastModified'],
                        checksum=obj['ETag'].strip('"'),
                        storage_class=obj.get('StorageClass', 'STANDARD'),
                    )
                    files.append(file_metadata)
            
            logger.info(f"Found {len(files)} files in s3://{self.bucket}/{prefix}")
            return files
        
        except ClientError as e:
            logger.error(f"Failed to list S3 objects: {e}")
            raise
    
    def file_exists(self, cloud_path: str) -> bool:
        """Check if a file exists in S3."""
        if not self._client:
            raise RuntimeError("Not connected to S3. Call connect() first.")
        
        try:
            self._client.head_object(Bucket=self.bucket, Key=cloud_path)
            return True
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                return False
            raise
    
    def get_file_metadata(self, cloud_path: str) -> Optional[FileMetadata]:
        """Get metadata for a file in S3."""
        if not self._client:
            raise RuntimeError("Not connected to S3. Call connect() first.")
        
        try:
            response = self._client.head_object(Bucket=self.bucket, Key=cloud_path)
            
            return FileMetadata(
                path=cloud_path,
                size=response['ContentLength'],
                modified_time=response['LastModified'],
                checksum=response['ETag'].strip('"'),
                storage_class=response.get('StorageClass', 'STANDARD'),
            )
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                return None
            raise
