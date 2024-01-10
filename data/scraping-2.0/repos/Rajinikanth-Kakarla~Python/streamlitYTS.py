import streamlit as st
from transformers import pipeline
from youtube_transcript_api import YouTubeTranscriptApi
import re
import openai

def get_video_id(url):
    parts = url.split("=")
    if len(parts) > 1:
        return parts[1]
    return None

st.title("YouTube Video Summarizer")

yt_video = st.text_input("Enter YouTube Video URL: [ex: https://www.youtube.com/watch?v=MS5UjNKw_1M]")
if yt_video:
    yt_vid = get_video_id(yt_video)

    st.video(yt_video)
    
    transcript = YouTubeTranscriptApi.get_transcript(yt_vid)
    
    result = ""
    for i in transcript:
        result += ' ' + i['text']

    summarizer = pipeline('summarization')
    num_iters = int(len(result) / 1000)
    sum_text = []
    for i in range(0, num_iters + 1):
        start = i * 1000
        end = (i + 1) * 1000
        out = summarizer(result[start:end])
        out = out[0]
        out = out['summary_text']
        sum_text.append(out)

    cleaned_text = re.sub(r'[^A-Za-z\s]+', '', str(sum_text))
    st.header("Summarized Text:")
    st.write(cleaned_text)

    text = cleaned_text

    summary = summarizer(text, max_length=150, min_length=30, do_sample=False)

    st.header("Corrected Text:")
    st.write(summary[0]["summary_text"])