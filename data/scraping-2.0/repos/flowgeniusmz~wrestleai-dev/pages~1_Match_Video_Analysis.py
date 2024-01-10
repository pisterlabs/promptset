import streamlit as st
from openai import OpenAI
import cv2
import base64
from tempfile import NamedTemporaryFile
import time
from config import pagesetup as ps

# Page Header
ps.set_title("WrestleAI", "Match Video Analysis")

# Section Header
ps.set_blue_header("Step 1: Upload Video")

video = st.file_uploader("Upload a video file", type=['.mp4', '.avi', '.mov', '.mkv'], accept_multiple_files=False)          # Get video

st.divider()

# Section Header

if video is not None:   
    ps.set_blue_header("Step 2: Create Temp File for Video")                                                                                                    # wait for upload
    with NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
        tfile.write(video.read()) 
        st.session_state.vidtempfile = tfile                                                                                          # write temp file 
        
        tfilepath = tfile.name     
        st.session_state.vidtempfilepath = tfilepath                                                                                        # get tempfile path
        
    videocapture = cv2.VideoCapture(tfilepath)

    videototalframes = int(videocapture.get(cv2.CAP_PROP_FRAME_COUNT))
    st.session_state.vidtotalframes = videototalframes

    ps.set_green_header("Video Dispaly")
    st.video(tfilepath)

    ps.set_green_header("Temporary File Path")
    st.markdown(f"**Temporary File Path:** {tfilepath}")

    ps.set_green_header("Frame Details")
    st.markdown(f"**Frames Read:** {videototalframes}")
    st.divider()
# Section Header
    ps.set_blue_header("Step 3: Frame Selection")

    currentframe = st.slider(label="View a specific frame", min_value=0, max_value=videototalframes-1, key="vidcurrentframe")

    videocapture.set(cv2.CAP_PROP_POS_FRAMES, currentframe)

    success, selectedframe = videocapture.read()
    if success:
        _, buffer = cv2.imencode('.jpg', selectedframe)
        selectedimage = st.image(buffer.tobytes(), channels="BGR")
    
    videocapture.release()