from dotenv import load_dotenv
from openai import OpenAI
import streamlit as st
import cv2  # pip install opencv-python
import base64
import tempfile
import os
import requests

if not load_dotenv():
    print("Could not load .env file or it is empty. Please check if it exists and is readable.")
    exit(1)

OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")

# Initialize the OpenAI client with the API key
client = OpenAI(api_key=OPENAI_API_KEY)

# Function to encode the image to base64
def encode_image(image_file):
    return base64.b64encode(image_file.getvalue()).decode("utf-8")

@st.cache_data
def video_to_base64_frames(video_buffer, frame_count=None, frame_interval=2):
    """Convert video to a series of base64 encoded frames.

    Args:
        video_buffer: The video file buffer.
        frame_count: The number of frames to process. If None, process all frames.
        frame_interval: The interval between frames to capture.
    """
    base64_frames = []
    
    # Read the file's bytes
    video_bytes = video_buffer.read()
    
    # Create a temporary file for the video
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(video_bytes)
        temp_video_name = temp_video.name
        
    # Load the video from the temporary file
    video = cv2.VideoCapture(temp_video_name)
    
    # Read each frame from the video and encode it as base64
    count = 0
    frame_index = 0
    while video.isOpened():
        success, frame = video.read()
        if not success or (frame_count is not None and count >= frame_count):
            break
        if frame_index % frame_interval == 0:
            _, buffer = cv2.imencode(".jpg", frame)
            base64_frames.append(base64.b64encode(buffer).decode("utf-8"))
            count += 1
        frame_index += 1
    
    video.release()
    
    # Clean up the temporary file
    try:
        os.remove(temp_video_name)
    except Exception as e:
        print(f"Error removing temporary file: {e}")
    return base64_frames


# Initialize Streamlit app
st.title("Turning Videos into Voiceovers using OpenAI models")
st.markdown(
    "#### [GPT-4 Vision](https://platform.openai.com/docs/guides/vision) and [TTS](https://platform.openai.com/docs/models/tts) APIs"
)


# Initialize session state variables
if "base64_frames" not in st.session_state:
    st.session_state.base64_frames = None
if "script" not in st.session_state:
    st.session_state.script = ""


# File uploader for video files
uploaded_video = st.file_uploader("Upload a video file", type=["mp4"])
if uploaded_video:
    with st.expander("Watch video", expanded=False):
        st.video(uploaded_video)
# Process video and generate script
if uploaded_video is not None:
    if st.button("Convert Video to Frames"):
        with st.spinner("Converting Video to Frames..."):
            # Convert video to base64 frames and store in session state
            st.session_state.base64_frames = video_to_base64_frames(uploaded_video)
            st.success(f"{len(st.session_state.base64_frames)} frames read.")
        # Display a sample frame from the video
        with st.expander("A Sample Frame", expanded=False):
            st.image(
                base64.b64decode(st.session_state.base64_frames[0].encode("utf-8")),
                caption="Sample Frame",
            )

# Button to generate script
if st.session_state.base64_frames and st.button("Generate Script"):
    PROMPT_MESSAGES = [
        {
            "role": "user",
            "content": [
                #"These are frames from a cooking show video. Generate a brief voiceover script in the style of a famous narrator, capturing the excitement and passion of holiday cooking. Only include the narration.",
                # "This is a frame from a golf swing video. Create a simple voice-over script in the style of a famous commentator to capture the posture and precautions of your swing. Include only commentary.",
                "These are frames from a video that I want to upload. Generate a compelling description that I can upload along with the video.",
                *map(
                    lambda x: {"image": x, "resize": 768},
                    st.session_state.base64_frames[0::50],
                ),
            ],
        },
    ]
    with st.spinner("Generating script..."):
        full_response = ""
        message_placeholder = st.empty()
        
        # Call OpenAI API to generate script based on the video frames
        for completion in client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=PROMPT_MESSAGES,
            max_tokens=1000,
            stream=True,
        ):
            # Check if there is content to display
            if completion.choices[0].delta.content is not None:
                full_response += completion.choices[0].delta.content
                message_placeholder.markdown(full_response + "▌")
            st.session_state.script = full_response
            
        with st.expander("Edit Generated Script:", expanded=False):
            st.text_area("Generated Script", st.session_state.script, height=250)

# Button to generate audio
if st.session_state.script and st.toggle("Generate Audio"):
    with st.spinner("Generating audio..."):
        response = requests.post(
            "https://api.openai.com/v1/audio/speech",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
            json={"model": "tts-1", "input": st.session_state.script, "voice": "fable"},
        )
        
        # Check the response status and handle audio generation
        if response.status_code == 200:
            audio_bytes = response.content
            if len(audio_bytes) > 0:
                # Temporary file creation for the audio
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                    fp.write(audio_bytes)
                    fp.seek(0)
                    st.audio(fp.name, format="audio/mp3")
                    with st.expander("Script", expanded=True):
                        st.write(st.session_state.script)
                    # Reset file pointer for download
                    fp.seek(0)
                    # Create a download button for the audio file
                    st.download_button(
                        label="Download audio",
                        data=fp.read(),
                        file_name="narration.mp3",
                        mime="audio/mp3",
                    )
                os.unlink(fp.name)  # Clean up the temporary file