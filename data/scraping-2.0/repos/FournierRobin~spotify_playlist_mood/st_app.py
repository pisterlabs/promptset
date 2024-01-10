# app.py

import streamlit as st
from langchain_utils import run_llm_on_text, parse_response
from spotify_utils import authenticate_to_spotify, get_songs_based_on_mood_data

# Streamlit Page Configuration
st.set_page_config(page_title="Spotify Mood Playlist Generator", page_icon="🎵", layout="wide")

st.title("Spotify Mood Playlist Generator")
st.write("Describe how you're feeling and get a playlist tailored to your mood!")

# User input
user_input = st.text_area("How are you feeling today?", "I'm feeling super energetic today and I'd like to listen to some music to get even better.")

# Button to generate playlist
if st.button("Generate Playlist"):
    # Get mood data from LLM and parse the response
    response = run_llm_on_text(user_input)
    mood_data = parse_response(response)
    
    # Authenticate to Spotify and get songs
    headers = authenticate_to_spotify()
    songs = get_songs_based_on_mood_data(mood_data, headers)
    
    # Display the recommended songs
    for song in songs:
        st.write(f"[{song['name']} by {song['artist']}]({song['url']})")
