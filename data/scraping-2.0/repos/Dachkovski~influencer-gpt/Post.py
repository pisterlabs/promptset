# Import necessary libraries
import os
import openai
from dotenv import load_dotenv
import tweepy
import streamlit as st
import json
from clients.d_id import DIdClient
from app.create_video_script import create_video_script
from app.get_trends import get_trends
from app.generate_video import generate_video
from app.upload_video import upload_video
import asyncio


# Load environment variables
load_dotenv()

# Authenticate Twitter API
bearer_token = st.session_state.get('TWITTER_BEARER_TOKEN', '')
twitter_client = tweepy.Client(bearer_token)

# Authenticate OpenAI API
openai.api_key = st.session_state.get('YOUR_OPENAI_API_KEY', '')

# Configure HeyGen API
HEYGEN_API_ENDPOINT = "https://api.heygen.com/v1/video.generate"
HEYGEN_API_KEY = st.session_state.get('HEYGEN_API_KEY', '')
avatar_id = os.getenv("YOUR_AVATAR_ID")  # Assuming this is still from an env variable

# Authenticate D-ID API
D_ID_API_KEY = st.session_state.get('D_ID_API_KEY', '')
d_id_client = DIdClient(api_key=D_ID_API_KEY)



async def check_video_status_async(d_id_client, talk_id):
    with st.spinner("Video is being processed..."):
        while not d_id_client.is_video_ready(talk_id):
            await asyncio.sleep(10)
    return d_id_client.get_video_url(talk_id)

    

def load_settings():
    # Load settings from file if it exists
    if os.path.exists("settings.json"):
        with open("settings.json", "r") as f:
            settings_data = json.load(f)
        
        for key, value in settings_data.items():
            st.session_state[key] = value


def main():
    load_settings()
    st.header("Influencer GPT :bird:")
    query = st.text_input("Trend Topic")

    if query:
        with st.spinner('Loading trends...'):
            trends = get_trends(query, st.session_state)

        # Let the user select a trend from the list
        selected_trend = st.radio("Select a trend", trends)

        with st.spinner('Generating script...'):
            # Use the selected trend to generate the script
            script = create_video_script(selected_trend)
        
        # Let the user edit the generated script
        edited_script = st.text_area("Edit the generated script:", script)
        
        # Button to confirm editing and proceed to video generation
        if st.button("Confirm and Generate Video"):
            # Use the edited script for video generation
            source_url = st.session_state.get('last_uploaded_image', '')
            video_url = generate_video(edited_script, source_url)
            upload_video(video_url)


if __name__ == '__main__':
    main()



