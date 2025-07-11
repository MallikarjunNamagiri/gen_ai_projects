# backend/workflow.py
import os
import time
import json
from typing import Dict, Optional
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from loguru import logger

# from backend.llm_processor import LLMProcessor
from .llm_processor import LLMProcessor
from backend.sd_processor import SDProcessor
from backend.video_processor import VideoProcessor
from backend.config import OUTPUT_DIR, DEFAULT_VIDEO_DURATION

class VideoWorkflow:
    def __init__(self):
        self.llm = LLMProcessor()
        self.sd = SDProcessor()
        self.video_processor = VideoProcessor()
        
    def run_workflow(self, workflow_data: Dict) -> Optional[str]:
        """Execute the complete video generation workflow"""
        try:
            start_time = time.time()
            
            # Step 1: Generate script
            logger.info("Generating script...")
            script_data = self._generate_script(
                workflow_data["prompt"],
                workflow_data.get("temperature", 0.7)
            )
            
            # Step 2: Generate images in parallel for each scene
            logger.info("Generating images...")
            image_paths = self._generate_images(script_data)
            
            # Step 3: Create video
            logger.info("Creating video...")
            video_path = self._create_video(script_data, image_paths)
            
            elapsed = time.time() - start_time
            logger.success(f"Workflow completed in {elapsed:.2f} seconds")
            
            return video_path
            
        except Exception as e:
            logger.error(f"Workflow failed: {str(e)}")
            return None
    
    def _generate_script(self, prompt: str, temperature: float) -> Dict:
        """Generate video script with LLM"""
        script_data = self.llm.generate_script(prompt, temperature)
        
        # Save script
        script_path = Path(OUTPUT_DIR) / "scripts" / f"script_{int(time.time())}.json"
        with open(script_path, 'w') as f:
            json.dump(script_data, f)
            
        return script_data
    
    def _generate_images(self, script_data: Dict) -> list:
        """Generate images for each scene in parallel"""
        image_prompts = [scene["image_prompt"] for scene in script_data["scenes"]]
        
        with ThreadPoolExecutor() as executor:
            image_paths = list(executor.map(
                lambda p: self.sd.generate_images([p])[0],
                image_prompts
            ))
            
        return image_paths
    
    def _create_video(self, script_data: Dict, image_paths: list) -> str:
        """Create final video from assets"""
        # Adjust scene durations based on total video duration
        total_duration = DEFAULT_VIDEO_DURATION
        scene_count = len(script_data["scenes"])
        duration_per_scene = total_duration / scene_count
        
        for scene in script_data["scenes"]:
            scene["duration"] = duration_per_scene
            
        # Generate video
        video_path = self.video_processor.create_video(
            image_paths,
            script_data,
            os.path.join(OUTPUT_DIR, "videos")
        )
        
        return video_path