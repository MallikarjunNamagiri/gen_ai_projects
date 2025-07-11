from moviepy.editor import ImageSequenceClip, AudioFileClip, concatenate_videoclips
from moviepy.video.fx.all import resize
import os
from config import DEFAULT_AUDIO_PATH

class VideoProcessor:
    def __init__(self):
        self.default_audio = DEFAULT_AUDIO_PATH
        
    def create_video(self, image_paths, script_data, output_path="outputs/videos"):
        os.makedirs(output_path, exist_ok=True)
        
        # Create clips for each scene
        clips = []
        for i, scene in enumerate(script_data["scenes"]):
            img_clip = ImageSequenceClip([image_paths[i]], durations=[scene["duration"]])
            img_clip = img_clip.set_position(("center", "center"))
            
            # Add text overlay if needed
            # (Implementation depends on your specific requirements)
            
            clips.append(img_clip)
        
        # Concatenate all clips
        final_clip = concatenate_videoclips(clips)
        
        # Add audio
        if os.path.exists(self.default_audio):
            audio = AudioFileClip(self.default_audio)
            final_clip = final_clip.set_audio(audio)
        
        # Write output
        output_file = f"{output_path}/video_{int(time.time())}.mp4"
        final_clip.write_videofile(output_file, fps=24, codec="libx264")
        
        return output_file