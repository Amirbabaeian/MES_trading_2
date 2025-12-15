"""
Custom exception classes for data I/O operations.

Provides specific error types for schema validation, file operations, and
data integrity issues.
"""


class ParquetIOError(Exception):
    """Base exception for all Parquet I/O operations."""
    pass


class SchemaValidationError(ParquetIOError):
    """
    Raised when data does not match the expected schema.
    
    Attributes:
        file_path: Path to the file that caused the error
        schema_error: Detailed description of schema mismatch
    """
    
    def __init__(
        self,
        message: str,
        file_path: str = None,
        schema_error: str = None
    ):
        """
        Initialize schema validation error.
        
        Args:
            message: Human-readable error message
            file_path: Optional path to problematic file
            schema_error: Optional detailed schema mismatch information
        """
        self.file_path = file_path
        self.schema_error = schema_error
        
        full_message = message
        if file_path:
            full_message += f"\n  File: {file_path}"
        if schema_error:
            full_message += f"\n  Details: {schema_error}"
        
        super().__init__(full_message)


class SchemaMismatchError(SchemaValidationError):
    """Raised when actual schema does not match expected schema."""
    
    def __init__(
        self,
        expected: dict,
        actual: dict,
        file_path: str = None,
        mismatches: list = None
    ):
        """
        Initialize schema mismatch error with comparison details.
        
        Args:
            expected: Expected schema as dict
            actual: Actual schema as dict
            file_path: Optional path to file
            mismatches: List of specific mismatches
        """
        self.expected = expected
        self.actual = actual
        self.mismatches = mismatches or []
        
        message = "Schema mismatch detected"
        if mismatches:
            mismatch_str = "\n  ".join(mismatches)
            schema_error = f"Column mismatches:\n  {mismatch_str}"
        else:
            schema_error = f"Expected: {expected}\nActual: {actual}"
        
        super().__init__(message, file_path, schema_error)


class MissingColumnsError(SchemaValidationError):
    """Raised when required columns are missing from the data."""
    
    def __init__(self, missing_columns: list, file_path: str = None):
        """
        Initialize missing columns error.
        
        Args:
            missing_columns: List of missing column names
            file_path: Optional path to file
        """
        self.missing_columns = missing_columns
        message = f"Missing required columns: {', '.join(missing_columns)}"
        super().__init__(message, file_path, None)


class DataTypeError(SchemaValidationError):
    """Raised when data types don't match schema."""
    
    def __init__(self, column: str, expected_type: str, actual_type: str, file_path: str = None):
        """
        Initialize data type error.
        
        Args:
            column: Column name
            expected_type: Expected data type
            actual_type: Actual data type
            file_path: Optional path to file
        """
        self.column = column
        self.expected_type = expected_type
        self.actual_type = actual_type
        message = f"Type mismatch in column '{column}'"
        schema_error = f"Expected {expected_type}, got {actual_type}"
        super().__init__(message, file_path, schema_error)


class ParquetReadError(ParquetIOError):
    """Raised when Parquet file reading fails."""
    
    def __init__(self, file_path: str, original_error: Exception = None):
        """
        Initialize read error.
        
        Args:
            file_path: Path to file that failed to read
            original_error: Original exception from pyarrow/pandas
        """
        self.file_path = file_path
        self.original_error = original_error
        
        message = f"Failed to read Parquet file: {file_path}"
        if original_error:
            message += f"\n  Reason: {str(original_error)}"
        
        super().__init__(message)


class ParquetWriteError(ParquetIOError):
    """Raised when Parquet file writing fails."""
    
    def __init__(self, file_path: str, original_error: Exception = None):
        """
        Initialize write error.
        
        Args:
            file_path: Path to file that failed to write
            original_error: Original exception from pyarrow/pandas
        """
        self.file_path = file_path
        self.original_error = original_error
        
        message = f"Failed to write Parquet file: {file_path}"
        if original_error:
            message += f"\n  Reason: {str(original_error)}"
        
        super().__init__(message)


class CorruptedFileError(ParquetIOError):
    """Raised when a Parquet file is corrupted or unreadable."""
    
    def __init__(self, file_path: str, reason: str = None):
        """
        Initialize corrupted file error.
        
        Args:
            file_path: Path to corrupted file
            reason: Reason for corruption
        """
        self.file_path = file_path
        self.reason = reason
        
        message = f"Corrupted or invalid Parquet file: {file_path}"
        if reason:
            message += f"\n  Reason: {reason}"
        
        super().__init__(message)


class CompressionError(ParquetIOError):
    """Raised when compression/decompression fails."""
    
    def __init__(self, compression: str, file_path: str = None):
        """
        Initialize compression error.
        
        Args:
            compression: Compression type that failed
            file_path: Optional path to file
        """
        self.compression = compression
        self.file_path = file_path
        
        message = f"Compression error: unsupported or invalid compression '{compression}'"
        if file_path:
            message += f"\n  File: {file_path}"
        
        super().__init__(message)


class PartitioningError(ParquetIOError):
    """Raised when partitioning operations fail."""
    
    def __init__(self, message: str, partition_key: str = None):
        """
        Initialize partitioning error.
        
        Args:
            message: Error message
            partition_key: Optional partition key that caused error
        """
        self.partition_key = partition_key
        
        full_message = message
        if partition_key:
            full_message += f"\n  Partition key: {partition_key}"
        
        super().__init__(full_message)
