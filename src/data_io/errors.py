"""
Custom exception classes for Parquet I/O and schema validation.
"""


class ParquetIOError(Exception):
    """Base exception for Parquet I/O errors."""
    pass


class SchemaMismatchError(ParquetIOError):
    """Raised when Parquet file schema doesn't match expected schema."""
    
    def __init__(self, message: str, expected_columns=None, actual_columns=None):
        """
        Initialize SchemaMismatchError.
        
        Args:
            message: Error message
            expected_columns: List of expected column names
            actual_columns: List of actual column names in file
        """
        super().__init__(message)
        self.expected_columns = expected_columns or []
        self.actual_columns = actual_columns or []
    
    def get_missing_columns(self):
        """Get columns that are expected but missing."""
        return set(self.expected_columns) - set(self.actual_columns)
    
    def get_extra_columns(self):
        """Get columns that exist but weren't expected."""
        return set(self.actual_columns) - set(self.expected_columns)


class MissingRequiredColumnError(ParquetIOError):
    """Raised when required columns are missing from the data."""
    
    def __init__(self, missing_columns):
        """
        Initialize MissingRequiredColumnError.
        
        Args:
            missing_columns: List of missing column names
        """
        self.missing_columns = missing_columns
        message = f"Missing required columns: {', '.join(missing_columns)}"
        super().__init__(message)


class DataTypeValidationError(ParquetIOError):
    """Raised when data types don't match schema."""
    
    def __init__(self, column_name: str, expected_type: str, actual_type: str):
        """
        Initialize DataTypeValidationError.
        
        Args:
            column_name: Name of the column with type mismatch
            expected_type: Expected data type
            actual_type: Actual data type in file
        """
        self.column_name = column_name
        self.expected_type = expected_type
        self.actual_type = actual_type
        message = f"Column '{column_name}': expected type '{expected_type}', got '{actual_type}'"
        super().__init__(message)


class CorruptedFileError(ParquetIOError):
    """Raised when Parquet file is corrupted or unreadable."""
    pass


class PartitioningError(ParquetIOError):
    """Raised when partitioned dataset operations fail."""
    pass


class CompressionError(ParquetIOError):
    """Raised when compression configuration is invalid."""
    
    def __init__(self, compression: str, valid_options):
        """
        Initialize CompressionError.
        
        Args:
            compression: Invalid compression codec
            valid_options: List of valid compression options
        """
        self.compression = compression
        self.valid_options = valid_options
        message = f"Invalid compression '{compression}'. Valid options: {', '.join(valid_options)}"
        super().__init__(message)


class MetadataError(ParquetIOError):
    """Raised when metadata operations fail."""
    pass
