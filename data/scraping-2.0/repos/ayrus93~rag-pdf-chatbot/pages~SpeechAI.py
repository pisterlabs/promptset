import streamlit as st
from streamlit_option_menu import option_menu
import openai


def auido_to_text(audio):
    openai.api_key = st.secrets["openai"]["OPENAI_API_KEY"]
    result = openai.Audio.transcribe("whisper-1", audio, verbose=True)
    return result


def video_to_text(video):
    openai.api_key = st.secrets["openai"]["OPENAI_API_KEY"]
    result = openai.Audio.transcribe("whisper-1", video, verbose=True)
    return result


selected = option_menu(
    menu_title=None,
    options=['Intro', 'Audio', 'Video'],
    icons=['menu-app', 'music-note', 'file-earmark-play'],
    menu_icon='cast',
    default_index=0,
    orientation='horizontal'
)
if selected == 'Intro':
    st.write("""
    🎧📜 Welcome to our Audio Transcription Web App! 🎤🔊

Are you looking for an efficient and accurate way to convert your audio content into text? Look no further! Our innovative web application powered by Python and OpenAI simplifies the process of transcribing audio into written form. Whether you're dealing with interviews, lectures, podcasts, or any audio content, our app streamlines the conversion, saving you valuable time and effort.

Key Features:

✨ Seamless Uploading: With a user-friendly interface, effortlessly upload your MP3/MP$ files directly from your local.

🎙️ Efficient Transcription: Our app harnesses the power of OpenAI, a cutting-edge transcription service. OpenAI leverages advanced machine learning algorithms to accurately transcribe spoken words into text, maintaining the nuances and subtleties of the audio.

📝 Instant Results: Experience the speed of modern technology! Within moments, you'll receive the transcribed text output on your screen, ready for review and further use.

How to Use:

Click the "Choose an Audio or Video " button.
Select your desired audio/video file from your local device.
Watch as our app works its magic and provides you with a written transcript.
""")
elif selected == 'Audio':
    st.empty()
    audio = st.file_uploader("Upload an audio file", type=["mp3"])
    if audio is not None:
        with st.spinner("Converting audio to speech...."):
            result = auido_to_text(audio)
        with st.container():
            st.write(result['text'])
            st.download_button(
                'Download the result as text file', result['text']
            )
elif selected == 'Video':
    st.empty()
    video = st.file_uploader("Upload video file", type=['mp4'])
    if video is not None:
        with st.spinner("Converting video to speech...."):
            result = video_to_text(video)
        with st.container():
            st.write(result['text'])
            st.download_button(
                'Download the result as text file', result['text']
            )