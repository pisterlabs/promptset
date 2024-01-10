import streamlit as st
import os
import streamlit as st
from dotenv import load_dotenv
from pytube.exceptions import VideoUnavailable
from moviepy.editor import *
from pytube import YouTube
import langchain as lc
import tempfile
import openai


from model_summary import generate_answer, generate_video_summary



def valid_url(url: str) -> bool:
    try:
        yt = YouTube(url)
        if not yt.video_id:
            return False
    except (VideoUnavailable, Exception):
        return False
    return yt.streams.filter(adaptive=True).first() is not None

def video_info(url: str):
    yt = YouTube(url)
    title = yt.title
    return title

st.set_page_config(page_title="YouTubeGPT")

os.environ["OPENAI_API_KEY"] = st.secrets['openai_key']
#  main app UI
def main():     
    st.header(":orange[YouTubeGPT Video Summarizer]")
    st.subheader(":orange[Discover video insights. Built with OpenAI & Langchain] 🚀")
    st.markdown('#') 
    choice = st.radio("Go ahead and make your selection:", ('Video Summary', 'Question-Answering'), horizontal=True)
    st.markdown('#') 

    openai_api_key = st.secrets['openai_key']

    # Enter yourtube URL
    youtube_url = st.text_input("Enter YouTube Video URL")

    
    if valid_url(youtube_url):
        video_title = video_info(youtube_url)
        st.markdown(f"##### {video_title}")
        st.video(youtube_url)
        
    else:
        st.error("Please enter a valid YouTube URL.")

    if choice == "Video Summary":
        if st.button("Summarize"):
            if not youtube_url:
                st.warning("Please enter a valid YouTube URL.")
            else:
                with st.spinner("Generating summary..."):
                    summary = generate_video_summary(openai_api_key, youtube_url)
                st.markdown(f"##### Summary of the Video:")
                st.success(summary)

    elif choice == "Question-Answering":
        if youtube_url:
            st.markdown('##### Would you like to explore further details regarding this video?')
            question = st.text_input("Submit your questions here.")
        else:
            st.markdown('##### Would you like to explore further details regarding this video?')
            question = st.text_input("Submit your questions here.", disabled=True)
            
        if st.button("Answer"):
            if not youtube_url:
                st.warning("Please enter a valid YouTube URL.")
            elif not question:
                st.warning("Please enter your question.")
            else:
                with st.spinner("Generating answer..."):
                    answer = generate_answer(openai_api_key, youtube_url, question)
                st.success(answer)

if __name__ == "__main__":
    main()
