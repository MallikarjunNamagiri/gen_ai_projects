import os
import json
import time
import requests
from typing import Dict, Optional
from pathlib import Path
import os
from pathlib import Path

# Directory configuration
BASE_DIR = Path(__file__).parent.parent
OUTPUT_DIR = BASE_DIR / "outputs"

# API configuration (use environment variables with defaults)
LLM_API_URL = os.getenv("LLM_API_URL", "http://localhost:8000")
SD_API_URL = os.getenv("SD_API_URL", "http://localhost:8000")
VIDEO_API_URL = os.getenv("VIDEO_API_URL", "http://localhost:8000")

# Model configuration
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "deepseek-ai/deepseek-llm-7b")
SD_MODEL_NAME = os.getenv("SD_MODEL_NAME", "runwayml/stable-diffusion-v1-5")

# Generation parameters
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "1024"))
DEFAULT_VIDEO_DURATION = int(os.getenv("DEFAULT_VIDEO_DURATION", "60"))  # seconds
DEFAULT_AUDIO_PATH = os.getenv("DEFAULT_AUDIO_PATH", str(BASE_DIR / "assets" / "background.mp3"))

# Performance settings
SD_SAFETY_CHECKER = os.getenv("SD_SAFETY_CHECKER", "True") == "True"
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
TIMEOUT = int(os.getenv("TIMEOUT", "30"))

def ensure_output_dirs():
    """Create necessary output directories if they don't exist"""
    (OUTPUT_DIR / "scripts").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "images").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "videos").mkdir(parents=True, exist_ok=True)

def generate_script(prompt: str, temperature: float = 0.7) -> Dict:
    """Generate video script using LLM API"""
    payload = {
        "prompt": prompt,
        "temperature": temperature,
        "max_length": 1024
    }
    
    response = requests.post(
        f"{LLM_API_URL}/generate-script",
        json=payload,
        timeout=30
    )
    response.raise_for_status()
    
    script_data = response.json()
    script_path = f"{OUTPUT_DIR}/scripts/script_{int(time.time())}.json"
    
    with open(script_path, 'w') as f:
        json.dump(script_data, f)
    
    return script_data

def generate_images(prompts: list) -> list:
    """Generate images using Stable Diffusion API"""
    payload = {
        "prompts": prompts,
        "output_dir": f"{OUTPUT_DIR}/images"
    }
    
    response = requests.post(
        f"{SD_API_URL}/generate-images",
        json=payload,
        timeout=60
    )
    response.raise_for_status()
    
    return response.json()["image_paths"]

def create_video(script_data: Dict, image_paths: list) -> str:
    """Create video from script and images"""
    payload = {
        "script_data": script_data,
        "image_paths": image_paths,
        "output_path": f"{OUTPUT_DIR}/videos"
    }
    
    response = requests.post(
        f"{VIDEO_API_URL}/create-video",
        json=payload,
        timeout=120
    )
    response.raise_for_status()
    
    return response.json()["video_path"]

def generate_video_workflow(workflow_data: Dict) -> Optional[str]:
    """Orchestrate the complete video generation workflow"""
    ensure_output_dirs()
    
    try:
        # Step 1: Generate script
        script_data = generate_script(
            workflow_data["prompt"],
            workflow_data.get("temperature", 0.7)
        )
        
        # Step 2: Generate images
        image_prompts = [scene["image_prompt"] for scene in script_data["scenes"]]
        image_paths = generate_images(image_prompts)
        
        # Step 3: Create video
        video_path = create_video(script_data, image_paths)
        
        return video_path
    
    except Exception as e:
        print(f"Workflow failed: {str(e)}")
        return None

def get_video_preview(video_path: str) -> Optional[str]:
    """Check if video file exists and return its path"""
    if video_path and os.path.exists(video_path):
        return video_path
    return None