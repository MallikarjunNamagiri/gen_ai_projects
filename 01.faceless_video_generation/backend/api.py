# backend/api.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

from backend.workflow import VideoWorkflow
from backend.llm_processor import LLMProcessor
from backend.sd_processor import SDProcessor
from backend.video_processor import VideoProcessor
from backend.config import OUTPUT_DIR


app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request models
class ScriptRequest(BaseModel):
    prompt: str
    temperature: float = 0.7

class ImageRequest(BaseModel):
    prompts: List[str]
    output_dir: str = None

class VideoRequest(BaseModel):
    script_data: dict
    image_paths: List[str]
    output_path: str = None

# Initialize processors
workflow = VideoWorkflow()
llm_processor = LLMProcessor()
sd_processor = SDProcessor()
video_processor = VideoProcessor()

@app.post("/generate-script")
async def generate_script(request: ScriptRequest):
    try:
        script_data = llm_processor.generate_script(
            request.prompt,
            request.temperature
        )
        return {"status": "success", "data": script_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-images")
async def generate_images(request: ImageRequest):
    try:
        image_paths = sd_processor.generate_images(
            request.prompts,
            request.output_dir or f"{OUTPUT_DIR}/images"
        )
        return {"status": "success", "image_paths": image_paths}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/create-video")
async def create_video(request: VideoRequest):
    try:
        video_path = video_processor.create_video(
            request.image_paths,
            request.script_data,
            request.output_path or f"{OUTPUT_DIR}/videos"
        )
        return {"status": "success", "video_path": video_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/run-workflow")
async def run_workflow(request: dict):
    try:
        video_path = workflow.run_workflow(request)
        if video_path:
            return {"status": "success", "video_path": video_path}
        else:
            raise Exception("Video generation failed")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))