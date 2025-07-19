import os
from dotenv import load_dotenv
from typing import List, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_environment():
    """Load environment variables from .env file"""
    load_dotenv()
    
    # Check for required environment variables
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError(
            "OPENAI_API_KEY not found in environment variables. "
            "Please create a .env file with your OpenAI API key."
        )
    
    return {
        "openai_api_key": openai_api_key
    }

def create_directories():
    """Create necessary directories for the application"""
    directories = ["documents", "vector_db"]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.info(f"Created directory: {directory}")

def get_supported_file_types() -> List[str]:
    """Return list of supported file extensions"""
    return [".pdf", ".txt", ".md", ".docx"]

def is_supported_file(filename: str) -> bool:
    """Check if file type is supported"""
    return any(filename.lower().endswith(ext) for ext in get_supported_file_types())

def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format"""
    if size_bytes == 0:
        return "0B"
    
    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f}{size_names[i]}"

def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe storage"""
    # Remove or replace problematic characters
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    return filename 