"""
Faceless Video Generator - Streamlit Frontend Module

This package contains all the Streamlit UI components and frontend utilities 
for the faceless video generation system.
"""

__version__ = "0.1.0"

# Import key components to make them available at package level
from .main import main as run_app
from .utils import (
    generate_video_workflow,
    get_video_preview
)

__all__ = [
    'run_app',
    'generate_video_workflow',
    'get_video_preview'
]