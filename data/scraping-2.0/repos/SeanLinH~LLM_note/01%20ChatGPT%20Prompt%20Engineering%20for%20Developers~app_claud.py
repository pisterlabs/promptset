from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
from dotenv import load_dotenv
import os 
import streamlit as st

load_dotenv()

anthropic = Anthropic(
    # defaults to os.environ.get("ANTHROPIC_API_KEY")
    api_key=os.getenv("API_CLAUD"),
)

prompt = st.text_input('士桓AI哥在此')

if prompt:
    # AI_PROMPT = "My name is Sean (林士桓), a dedicated AI engineer from Taiwan, currently serving in a PCB manufacturing. I possess profound expertise and passion in AI, encompassing training, deployment, and application. My competencies extend to image processing, object recognition, YOLO, deep learning, machine learning, OpenCV, PyTorch, CNN, feature extraction, image segmentation, and model fine-tuning. As a 30-year-old professional, I am not only enthusiastic about embracing new challenges but also consistently expanding my network. Currently, I am actively seeking opportunities to transition to an overseas company, aspiring to further enhance my professional capabilities and career progression."
    completion = anthropic.completions.create(
        model="claude-2.1",
        max_tokens_to_sample=300,
        prompt=f"{HUMAN_PROMPT} {prompt} {AI_PROMPT}",
    )
    st.write(completion.completion)