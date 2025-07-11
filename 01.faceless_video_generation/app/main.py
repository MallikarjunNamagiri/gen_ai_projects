import streamlit as st
import os
import json
import time
from utils import generate_video_workflow, get_video_preview

# Configure page
st.set_page_config(page_title="Faceless Video Generator", layout="wide")

# Custom CSS
def load_css():
    with open("assets/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# Main app
def main():
    st.title("🎥 Faceless Video Generator")
    st.markdown("Create AI-powered videos from text prompts - no face or voice needed!")
    
    with st.sidebar:
        st.header("Settings")
        video_duration = st.selectbox("Video Duration", ["30s", "60s", "90s"], index=1)
        video_style = st.selectbox("Video Style", ["Minimal", "Infographic", "Storytelling", "Educational"], index=2)
        voice_model = st.selectbox("Voice Model", ["Male Narrator", "Female Narrator", "Child Voice"], index=0)
    
    # Main input form
    with st.form("video_input_form"):
        prompt = st.text_area("Enter your video topic or script idea:", 
                            placeholder="What if the moon suddenly disappeared?")
        
        col1, col2 = st.columns(2)
        with col1:
            generate_script = st.checkbox("Generate script with AI", value=True)
        with col2:
            advanced_options = st.checkbox("Show advanced options")
        
        if advanced_options:
            temperature = st.slider("Creativity level", 0.0, 1.0, 0.7)
            num_images = st.slider("Number of scenes", 3, 10, 5)
        
        submitted = st.form_submit_button("Generate Video")
    
    if submitted:
        if not prompt:
            st.warning("Please enter a video topic")
            return
            
        with st.spinner("Creating your video..."):
            # Prepare workflow data
            workflow_data = {
                "prompt": prompt,
                "generate_script": generate_script,
                "duration": video_duration,
                "style": video_style,
                "voice": voice_model,
                "temperature": temperature if advanced_options else 0.7,
                "num_images": num_images if advanced_options else 5
            }
            
            # Trigger n8n workflow
            workflow_id = generate_video_workflow(workflow_data)
            
            # Poll for completion
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i in range(10):
                progress_bar.progress((i + 1) * 10)
                status_text.text(f"Processing... Step {i+1} of 10")
                time.sleep(1)
            
            # Get results
            video_url = get_video_preview(workflow_id)
            
            if video_url:
                st.success("Video generated successfully!")
                st.video(video_url)
                
                # Download button
                with open(video_url, "rb") as f:
                    st.download_button(
                        label="Download Video",
                        data=f,
                        file_name="generated_video.mp4",
                        mime="video/mp4"
                    )
            else:
                st.error("Failed to generate video. Please try again.")

if __name__ == "__main__":
    main()