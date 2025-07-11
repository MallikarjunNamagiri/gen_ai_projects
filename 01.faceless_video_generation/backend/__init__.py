"""
Faceless Video Generator - Backend Processing Module
"""

__version__ = "0.1.0"

# Import config items directly without circular reference
from .config import (
    BASE_DIR,
    OUTPUT_DIR,
    LLM_API_URL,
    SD_API_URL,
    VIDEO_API_URL,
    LLM_MODEL_NAME,
    SD_MODEL_NAME,
    DEFAULT_AUDIO_PATH
)

# Lazy import processors to avoid circular imports
def get_llm_processor():
    from .llm_processor import LLMProcessor
    return LLMProcessor()

def get_sd_processor():
    from .sd_processor import SDProcessor
    return SDProcessor()

def get_video_processor():
    from .video_processor import VideoProcessor
    return VideoProcessor()

__all__ = [
    'BASE_DIR',
    'OUTPUT_DIR',
    'LLM_API_URL',
    'SD_API_URL',
    'VIDEO_API_URL',
    'LLM_MODEL_NAME',
    'SD_MODEL_NAME',
    'DEFAULT_AUDIO_PATH',
    'get_llm_processor',
    'get_sd_processor',
    'get_video_processor'
]