from transformers import AutoModelForCausalLM, AutoTokenizer
from .config import LLM_MODEL_NAME, MAX_NEW_TOKENS  # Changed to relative import

class LLMProcessor:
    def __init__(self):
        self.model = AutoModelForCausalLM.from_pretrained(LLM_MODEL_NAME)
        self.tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
        
    def generate_script(self, prompt, temperature=0.7, max_length=1024):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(
            inputs.input_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=temperature,
            do_sample=True,
            top_p=0.9
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def _format_script(self, raw_text):
        # Split into scenes
        scenes = [s.strip() for s in raw_text.split("\n\n") if s.strip()]
        
        # Format as JSON structure for video generation
        script_data = {
            "title": scenes[0] if scenes else "Generated Video",
            "scenes": []
        }
        
        for i, scene in enumerate(scenes[1:], 1):
            script_data["scenes"].append({
                "scene_number": i,
                "text": scene,
                "duration": 5,  # seconds per scene
                "image_prompt": scene[:100]  # Use first 100 chars as SD prompt
            })
            
        return script_data