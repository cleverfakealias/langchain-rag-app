import os
from typing import List, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_DIRECTORIES = ["documents", "vector_db"]
SUPPORTED_FILE_EXTENSIONS = [".pdf", ".txt", ".md", ".docx"]
FILE_SIZE_UNITS = ["B", "KB", "MB", "GB"]
INVALID_FILENAME_CHARS = '<>:"/\\|?*'
FILE_SIZE_BASE = 1024.0

def create_directories():
    """Create necessary directories for the application"""
    for directory in DEFAULT_DIRECTORIES:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.info(f"Created directory: {directory}")

def get_supported_file_types() -> List[str]:
    """Return list of supported file extensions"""
    return SUPPORTED_FILE_EXTENSIONS

def is_supported_file(filename: str) -> bool:
    """Check if file type is supported"""
    return any(filename.lower().endswith(ext) for ext in get_supported_file_types())

def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format"""
    if size_bytes == 0:
        return "0B"
    
    i = 0
    while size_bytes >= FILE_SIZE_BASE and i < len(FILE_SIZE_UNITS) - 1:
        size_bytes /= FILE_SIZE_BASE
        i += 1
    
    return f"{size_bytes:.1f}{FILE_SIZE_UNITS[i]}"

def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe storage"""
    # Remove or replace problematic characters
    for char in INVALID_FILENAME_CHARS:
        filename = filename.replace(char, '_')
    return filename 