from setuptools import setup, find_packages

setup(
    name="faceless-video-generator",
    version="0.2.0",
    packages=find_packages(),
    install_requires=[
        'torch>=2.0.0',
        'transformers>=4.30.0',
        'diffusers>=0.19.0',
        'moviepy>=1.0.0',
        'streamlit>=1.22.0',
        'loguru>=0.7.0',
    ],
    python_requires='>=3.8',
)