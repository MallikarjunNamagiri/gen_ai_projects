from diffusers import StableDiffusionPipeline
import torch
from config import SD_MODEL_NAME, SD_SAFETY_CHECKER

class SDProcessor:
    def __init__(self):
        self.pipe = StableDiffusionPipeline.from_pretrained(
            SD_MODEL_NAME,
            safety_checker=SD_SAFETY_CHECKER,
            torch_dtype=torch.float16
        )
        self.pipe = self.pipe.to("cuda" if torch.cuda.is_available() else "cpu")
        
    def generate_images(self, prompts, output_dir="outputs/images"):
        os.makedirs(output_dir, exist_ok=True)
        image_paths = []
        
        for i, prompt in enumerate(prompts):
            image = self.pipe(prompt).images[0]
            path = f"{output_dir}/scene_{i+1}.png"
            image.save(path)
            image_paths.append(path)
            
        return image_paths