import streamlit as st
import av
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoTokenizer, VisionEncoderDecoderModel
from pytube import YouTube
import os
import openai
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Check if CUDA is available and set the device accordingly
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load pretrained processor, tokenizer, and model
image_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = VisionEncoderDecoderModel.from_pretrained(
    "Neleac/timesformer-gpt2-video-captioning"
).to(device)


def main():
    hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)
    st.title("Video Captioning with TimesFormer-GPT2")

    # Option to choose between uploading a video file or pasting a YouTube link
    option = st.radio(
        "Choose an option:", ("Upload a video file", "Paste a YouTube link")
    )

    if option == "Upload a video file":
        video_file = st.file_uploader("Upload a video file", type=["mp4"])
        if video_file is not None:
            st.video(video_file)
            frames, seg_len = extract_frames(video_file)
            if frames:
                generate_captions(frames, seg_len)
    elif option == "Paste a YouTube link":
        youtube_link = st.text_input("Paste a YouTube link")
        if youtube_link:
            st.write("Downloading video from YouTube...")
            video_file_path = download_youtube_video(youtube_link)
            if video_file_path:
                st.video(video_file_path)
                frames, seg_len = extract_frames(video_file_path)
                if frames:
                    generate_captions(frames, seg_len)


# Add a function to download a YouTube video
def download_youtube_video(youtube_link):
    try:
        yt = YouTube(youtube_link)
        stream = yt.streams.filter(file_extension="mp4", progressive=True).first()
        video_file_path = stream.download()
        return video_file_path
    except Exception as e:
        st.error(f"Error downloading YouTube video: {e}")
        return None


# Add a function to extract frames from a video
def extract_frames(video_file):
    container = av.open(video_file)
    seg_len = container.streams.video[0].frames
    clip_len = model.config.encoder.num_frames
    indices = set(
        np.linspace(0, seg_len, num=clip_len, endpoint=False).astype(np.int64)
    )
    frames = []

    container.seek(0)
    for i, frame in enumerate(container.decode(video=0)):
        if i in indices:
            frames.append(frame.to_ndarray(format="rgb24"))

    return frames, seg_len


# Add a function to generate captions
def generate_captions(frames, seg_len):
    st.write("Generating captions...")

    gen_kwargs = {
        "min_length": 10,
        "max_length": 20,
        "num_beams": 8,
    }
    pixel_values = image_processor(frames, return_tensors="pt").pixel_values.to(device)
    tokens = model.generate(pixel_values, **gen_kwargs)
    caption = tokenizer.batch_decode(tokens, skip_special_tokens=True)[0]

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are given a caption line from a video summarizer model. Expand it into a few more lines. Don't get too creative to avoid writing something that is not in the video"},
            {"role": "user", "content": f"Caption: {caption}"},
        ],
    )

    st.write("Generated Caption:")
    if not completion:
        st.write(caption)
    else:
        st.write(completion.choices[0].message["content"])


if __name__ == "__main__":
    main()
