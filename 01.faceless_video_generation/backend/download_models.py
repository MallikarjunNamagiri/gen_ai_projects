import os
import requests
import shutil
import subprocess
from pathlib import Path
from huggingface_hub import snapshot_download, login
from loguru import logger
from backend.config import (
    LLM_MODEL_NAME,
    SD_MODEL_NAME,
    BASE_DIR,
    OUTPUT_DIR
)

# Configure logger
logger.add(
    os.path.join(BASE_DIR, "logs", "model_download.log"),
    rotation="10 MB",
    retention="7 days",
    level="INFO"
)

def download_llm_model():
    """Download the LLM model from Hugging Face Hub"""
    try:
        logger.info(f"Downloading LLM model: {LLM_MODEL_NAME}")
        
        model_dir = os.path.join(BASE_DIR, "models", "llm")
        os.makedirs(model_dir, exist_ok=True)
        
        # Use huggingface_hub to download the model
        snapshot_download(
            repo_id=LLM_MODEL_NAME,
            local_dir=model_dir,
            resume_download=True,
            allow_patterns=["*.json", "*.bin", "*.model", "*.py"],
            ignore_patterns=["*.h5", "*.tflite", "*.msgpack"],
            cache_dir=os.path.join(BASE_DIR, "cache")
        )
        
        logger.success(f"LLM model downloaded to {model_dir}")
        return True
    
    except Exception as e:
        logger.error(f"Failed to download LLM model: {str(e)}")
        return False

def download_sd_model():
    """Download the Stable Diffusion model from Hugging Face Hub"""
    try:
        logger.info(f"Downloading Stable Diffusion model: {SD_MODEL_NAME}")
        
        model_dir = os.path.join(BASE_DIR, "models", "stable-diffusion")
        os.makedirs(model_dir, exist_ok=True)
        
        # Use huggingface_hub to download the model
        snapshot_download(
            repo_id=SD_MODEL_NAME,
            local_dir=model_dir,
            resume_download=True,
            allow_patterns=["*.bin", "*.json", "*.pt", "*.ckpt", "*.safetensors"],
            ignore_patterns=["*.onnx", "*.tflite"],
            cache_dir=os.path.join(BASE_DIR, "cache")
        )
        
        logger.success(f"Stable Diffusion model downloaded to {model_dir}")
        return True
    
    except Exception as e:
        logger.error(f"Failed to download Stable Diffusion model: {str(e)}")
        return False

def download_background_audio():
    """Download default background audio if not present"""
    try:
        audio_dir = os.path.join(BASE_DIR, "assets")
        os.makedirs(audio_dir, exist_ok=True)
        audio_path = os.path.join(audio_dir, "background.mp3")
        
        if not os.path.exists(audio_path):
            logger.info("Downloading background audio...")
            
            # Example audio file (replace with your own or use a different source)
            audio_url = "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3"
            
            response = requests.get(audio_url, stream=True)
            response.raise_for_status()
            
            with open(audio_path, 'wb') as f:
                shutil.copyfileobj(response.raw, f)
            
            logger.success(f"Background audio downloaded to {audio_path}")
        
        return True
    
    except Exception as e:
        logger.error(f"Failed to download background audio: {str(e)}")
        return False

def check_disk_space():
    """Verify sufficient disk space is available"""
    try:
        # Require at least 20GB free space for models
        min_space = 20 * 1024 * 1024 * 1024  # 20GB in bytes
        stat = shutil.disk_usage(BASE_DIR)
        
        if stat.free < min_space:
            logger.warning(f"Insufficient disk space. Required: {min_space/1024**3:.1f}GB, Available: {stat.free/1024**3:.1f}GB")
            return False
        
        return True
    
    except Exception as e:
        logger.error(f"Failed to check disk space: {str(e)}")
        return False

def install_dependencies():
    """Install required Python dependencies"""
    try:
        logger.info("Installing Python dependencies...")
        
        requirements_path = os.path.join(BASE_DIR, "requirements.txt")
        subprocess.run(
            ["pip", "install", "-r", requirements_path],
            check=True
        )
        
        # Modified GPU detection to be more robust
        try:
            import torch
            if torch.cuda.is_available():
                logger.info("GPU detected - installing GPU-accelerated PyTorch")
                subprocess.run(
                    ["pip", "install", "torch", "torchvision", "torchaudio", 
                     "--index-url", "https://download.pytorch.org/whl/cu118"],
                    check=True
                )
        except ImportError:
            logger.warning("PyTorch not available - will install CPU version")
        
        logger.success("Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install dependencies: {str(e)}")
        return False

def main():
    """Main function to download all required models"""
    # Create necessary directories
    os.makedirs(os.path.join(BASE_DIR, "logs"), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "cache"), exist_ok=True)
    
    logger.info("Starting model download process...")
    
    # Check system requirements first
    if not check_disk_space():
        logger.error("Insufficient disk space. Please free up space and try again.")
        return
    
    # Install dependencies
    if not install_dependencies():
        logger.error("Failed to install dependencies. Check logs for details.")
        return
    
    # Download models
    success = True
    success &= download_llm_model()
    success &= download_sd_model()
    success &= download_background_audio()
    
    if success:
        logger.success("All models downloaded successfully!")
    else:
        logger.error("Some models failed to download. Check logs for details.")

if __name__ == "__main__":
    # Optional: Add your Hugging Face token if needed
    # from huggingface_hub import login
    # login(token="your_huggingface_token")
    
    main()