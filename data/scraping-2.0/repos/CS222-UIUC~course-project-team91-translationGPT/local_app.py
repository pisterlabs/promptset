import os
import subprocess
from pathlib import Path
import streamlit as st
import shutil
import base64
import re
import openai

def create_download_link(file_path, file_name):
    with open(file_path, "rb") as f:
        bytes = f.read()
        b64 = base64.b64encode(bytes).decode()
        href = f'<a href="data:file/octet-stream;base64,{b64}" download="{file_name}">Download {file_name}</a>'
        st.markdown(href, unsafe_allow_html=True)

st.set_page_config(
    page_title="Video Translator",
    page_icon=":globe_with_meridians:",
    layout="centered",
)

if not os.path.exists("streamlit_output"):
    os.mkdir("streamlit_output")

st.title("PigeonAI Video Translator")
st.subheader("Translate your video subtitles from English to Chinese\n 自动将视频字幕从英文翻译成中文")

inputs_count = 0
video_name = None

# YouTube video link input
video_link = st.text_input("Enter YouTube video link")
if video_link:
    inputs_count += 1

# Upload video file
video_file = st.file_uploader("Upload your video file", type=["mp4"])
if video_file:
    inputs_count += 1

# Upload audio file
audio_file = st.file_uploader("Upload your audio file (optional)", type=["mp3", "wav","m4a"])
if audio_file:
    inputs_count += 1

# Upload SRT file
srt_file = st.file_uploader("Upload your SRT file (optional)", type=["srt"])
if srt_file:
    inputs_count += 1

# check if the user wants to encode the video file
is_encoding = st.checkbox("Encode the video file (optional)", value=False)

# Check only one of the inputs is given
if inputs_count == 1:
    st.success("Valid input. Proceed with your operations.")
elif inputs_count > 1:
    st.error("Please provide only one input.")
else:
    st.warning("Please provide an input to proceed.")

if video_file is not None or video_link is not None or srt_file is not None or audio_file is not None:
    
    video_name = st.text_input("Enter a name for the output video", value="No need for this if you give a link")
    video_name = re.sub(r'\s+', '_', video_name)

    if st.button("Start Translation"):
        with st.spinner("Translating..."):
            video_file_path = Path(video_file.name) if video_file else None
            audio_file_path = Path(audio_file.name) if audio_file else None
            srt_file_path = Path(srt_file.name) if srt_file else None

            if video_file:
                with open(video_file_path, "wb") as f:
                    f.write(video_file.getbuffer())

            if audio_file:
                with open(audio_file_path, "wb") as f:
                    f.write(audio_file.getbuffer())

            if srt_file:
                with open(srt_file_path, "wb") as f:
                    f.write(srt_file.getbuffer())

            # temp_result_dir = tempfile.mkdtemp()
            
            # Save the paths to the input files as command line arguments
            cmd_args = [
                "python",
                "pipeline.py",
            ]

            if video_file:
                cmd_args.extend(["--video_file", f"'{video_file_path}'"])
                if is_encoding:
                    cmd_args.append("-v")

            if audio_file:
                cmd_args.extend(["--audio_file", f"'{audio_file_path}'"])

            if srt_file:
                cmd_args.extend(["--srt_file", f"'{srt_file_path}'"])

            if video_link:
                cmd_args.extend(["--link", video_link])
                if is_encoding:
                    cmd_args.append("-v")

            if video_name:
                cmd_args.extend(["--video_name", f"'{video_name}'"])
            
            
            cmd_args.extend([
                "--output_dir", "streamlit_output"
            ])

            cmd = " ".join(cmd_args)

            # Run the translation script
            result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            # st.write(result.stderr.decode("utf-8"))
            print(result.stdout.decode("utf-8"))
            print(result.returncode)
            print(result)

            if result.returncode == 0:
                st.success("Translation complete!")

                srt_file_zh = f"streamlit_output/{video_name}/{video_name}_zh.srt"
                ass_file_zh = f"streamlit_output/{video_name}/{video_name}_zh.ass"
                video_file_zh = f"streamlit_output/{video_name}/{video_name}.mp4"

                st.markdown("### Download Files")
                create_download_link(srt_file_zh, f"{video_name}_zh.srt")
                create_download_link(ass_file_zh, f"{video_name}_zh.ass")

                if os.path.exists(video_file_zh):
                    create_download_link(video_file_zh, f"{video_name}_zh.mp4")
                else:
                    st.warning("Translated video not available. Check your input settings.")
        
                # Clean up temporary files and directories
                shutil.rmtree("streamlit_output")
                if video_file:
                    os.remove(video_file_path)
                if audio_file:
                    os.remove(audio_file_path)
                if srt_file:
                    os.remove(srt_file_path)
            else:
                st.error("Translation failed. Please check the input settings and try again.")

