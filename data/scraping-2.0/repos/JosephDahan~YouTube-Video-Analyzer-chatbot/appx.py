import openai
import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
import requests
from urllib.parse import urlparse, parse_qs
import sys
from pytube import YouTube


def extract_video_id(url):
    """
    Extract the video ID from a YouTube URL.
    """
    # Parse the URL
    parsed_url = urlparse(url)
    
    # Get the video ID from the query parameters (for URLs with "v=")
    video_id = parse_qs(parsed_url.query).get("v")
    if video_id:
        return video_id[0]
    
    # Get the video ID from the URL path (for URLs with "youtu.be")
    if "youtu.be" in parsed_url.netloc:
        return parsed_url.path.split("/")[1]
    
    return None


# Add a "New Chat" button in the sidebar
if st.sidebar.button('New Chat'):
    st.session_state["url_entered"] = False
    st.session_state["summary_generated"] = False
    st.session_state["video_url"] = ""
    st.session_state["video_details"] = {}
    st.session_state["full_transcript"] = ""
    st.session_state.messages = []
    st.experimental_rerun()

# Prompt the user to enter their API keys in the sidebar
st.sidebar.title("API Key Configuration")
openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
#youtube_api_key = st.sidebar.text_input("YouTube API Key", type="password")



# Use the API keys in your app (only if they are provided)
if openai_api_key:
    openai.api_key = openai_api_key
else:
    st.sidebar.warning("Please enter your API keys to use the app.")

st.title("ChatGPT-like clone with YouTube Video Analyzer")

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo-16k"

if "messages" not in st.session_state:
    st.session_state.messages = []

if "video_url" not in st.session_state:
    st.session_state["video_url"] = ""

if "video_details" not in st.session_state:
    st.session_state["video_details"] = {}

if "full_transcript" not in st.session_state:
    st.session_state["full_transcript"] = ""

if "url_entered" not in st.session_state:
    st.session_state["url_entered"] = False

if "summary_generated" not in st.session_state:
    st.session_state["summary_generated"] = False

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Define a variable to hold the API response
api_response = None


if not st.session_state["url_entered"]:
    prompt = st.chat_input("Enter YouTube Video URL:")
    if prompt:
        video_id = extract_video_id(prompt)
        if not video_id:
            st.error("Invalid YouTube URL. Please check the URL and try again.")
            sys.exit()  # Use sys.exit() instead of return

        try:
            yt = YouTube(prompt)
            video_title = yt.title
            channel_name = yt.author
            
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            full_transcript = " ".join([x['text'] for x in transcript])

            st.session_state['video_details'] = {'items': [{'snippet': {'title': video_title, 'channelTitle': channel_name}}]}
            st.session_state['full_transcript'] = full_transcript
            st.session_state["url_entered"] = True
            st.success('Transcript fetched successfully!')
        except YouTubeTranscriptApi.TranscriptsDisabled:
            st.error('Could not retrieve transcript. The video might not have transcripts enabled.')
        except YouTubeTranscriptApi.NoTranscriptFound:
            st.error('Could not retrieve transcript. No transcript found.')
        except Exception as e:
            st.error(f'Error fetching transcript: {str(e)}')

        if not st.session_state.get('summary_generated', False) and openai_api_key:
            video_details = st.session_state['video_details']
            video_title = video_details['items'][0]['snippet']['title']
            full_transcript = st.session_state['full_transcript']
            
            st.session_state['current_chat'] = [
                {"role": "system", "content": 'You are a helpful assistant.'},
                {"role": "user", "content": f"I have watched a video titled '{video_title}' from the channel '{video_details['items'][0]['snippet']['channelTitle']}' with the following transcript: {full_transcript}"},
                {"role": "user", "content": "Can you give me a summary or gist of what the video is about? if needed, you can use points and elaborate quickly upon each point."}
            ]

            with st.spinner('Generating summary...'):
                try:
                    with st.chat_message("assistant"):
                        summary_message_placeholder = st.empty()
                        full_response = ""
                        for response in openai.ChatCompletion.create(
                            model=st.session_state["openai_model"], 
                            messages=st.session_state['current_chat'],
                            stream=True,
                        ):
                            full_response += response.choices[0].delta.get("content", "")
                            summary_message_placeholder.markdown(full_response + "▌")
                        summary_message_placeholder.markdown(full_response)
                    
                    st.session_state.messages.append({"role": "user", "content": st.session_state['current_chat'][2]["content"]})
                    st.session_state.messages.append({"role": "assistant", "content": full_response.strip()})
                    st.session_state['summary_generated'] = True
                    st.experimental_rerun()
                except openai.error.OpenAIError as e:
                    st.error(f'Error generating summary: {str(e)}')
else:
    if st.checkbox('Show Behind the Scenes - Transcript', key='transcript_behind_the_scenes'):
        st.write(f"Transcript: {st.session_state.get('full_transcript', '...')}")

    video_details = st.session_state['video_details']
    video_title = video_details['items'][0]['snippet']['title']
    channel_name = video_details['items'][0]['snippet']['channelTitle']
    full_transcript = st.session_state['full_transcript']

    prompt = st.chat_input(f"Ask something about '{video_title}' by {channel_name}:")
    if prompt and openai_api_key:
        user_message = f"I watched a video titled '{video_title}' from the channel '{channel_name}' with the following transcript: {full_transcript}. My question is: {prompt}"
        st.session_state.messages.append({"role": "user", "content": user_message})
        st.chat_message("user").write(prompt)  # Display the user's question

        with st.spinner('Generating response...'):
            try:
                with st.chat_message("assistant"):
                    assistant_message_placeholder = st.empty()
                    full_response = ""
                    messages_for_assistant = [
                        {"role": "system", "content": 'You are a helpful assistant.'},
                    ] + st.session_state.messages

                    for response in openai.ChatCompletion.create(
                        model=st.session_state["openai_model"],
                        messages=messages_for_assistant,
                        stream=True,
                    ):
                        full_response += response.choices[0].delta.get("content", "")
                        assistant_message_placeholder.markdown(full_response + "▌")
                    assistant_message_placeholder.markdown(full_response)

                st.session_state.messages.append({"role": "assistant", "content": full_response.strip()})
            except openai.error.OpenAIError as e:
                st.error(f'Error generating response: {str(e)}')