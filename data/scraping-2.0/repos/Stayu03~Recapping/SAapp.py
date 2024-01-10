import streamlit as st
import pandas as pd
import openai #importOpenAI 

# Set Streamlit app title and description
st.title("Music Mood Recommender 🎵")
st.write("Tell us about your mood! We will recommend 3 songs and 1 album based on your mood!")

# Add a sidebar for the OpenAI API key input
api_key_1 = st.sidebar.text_input("Enter your OpenAI API key", type="password", key="api_key_input")

# Set OpenAI API key
client = openai.OpenAI(api_key=api_key_1)
# Function to recommend songs and albums based on mood and language

def recommend_songs_and_album(mood, language):
    song_recommendations = []

    prompt_song1 = f"""
    Could you recommend the first song that expresses {mood} in {language}?
    I want details about the song.
    Could you tell me the artist name, album name, and a quote from the lyrics (3 sentences)?

    Could you give me information by this format for Example of the response:
    For the first song, I recommend the song "song name" by artist name. Could yougive me the song detail?
    *skip a line*
    Album: album name *skip a line*
    Quote from the lyrics: quote from the lyrics and translate it to English
    """

    prompt_song2 = f"""
    Could you recommend another song (not the same as prompt_song1) that expresses {mood} in {language} which is not the same artist as prompt_song1 ?
    I want details about the song.
    Could you tell me the artist name, album name, and a quote from the lyrics (3 sentences)?

    Could you give me information by this format for Example of the response:
    For the second song, I recommend the song "song name" by artist name. Could yougive me the song detail?
    *skip a line*
    Album: album name *skip a line*
    Quote from the lyrics: quote from the lyrics and translate it to English
    """

    prompt_song3 = f"""
    Could you recommend another song (not the same as prompt_song1 and prompt_song2) that expresses {mood} in {language} which is not the same as artist as prompt_song1 and prompt_song2?

    I want details about the song.
    Could you tell me the artist name, album name, and a quote from the lyrics (3 sentences)?

    Could you give me information by this format for Example of the response:
    For the third song, I recommend the song "song name" by artist name. Could yougive me the song detail?
    *skip a line*
    Album: album name *skip a line*
    Quote from the lyrics: quote from the lyrics and translate it to English
    """

    prompt_album = f"""
    You want to find albums that express a {mood} mood in {language}. 
    Could you recommend one album with its respective artist, release year, 
    3 songs from the album (including the title track) and qoute some lyrics from each songs?
    Please skip a line for each detail.

    For example of the response:
    The album that expresses a {mood} mood in {language} is "album name" by artist name. 
    Here are the additional details about the album! <skip a line>
    Artist : artist name <skpi a line>
    Release Year : release year <skip a line>
    3 Songs from the album and real lyrics from each songs: *new line*
    1. "song's name" - example of a quote from the lyrics and translate it to English *new line*
    2. "song's name" - example of a quote from the lyrics and translate it to English *new line*
    3. "song's name" - example of a quote from the lyrics and translate it to English *new line*
    note: only 3 sentences for each song
    """

    song_response1 = client.chat.completions.create(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt_song1}
    ],
    model="gpt-3.5-turbo",
    max_tokens=300,
    )

    song_response2 = client.chat.completions.create(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt_song2}
    ],  
    model="gpt-3.5-turbo",
    max_tokens=300
    )

    song_response3 = client.chat.completions.create(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt_song3}
    ],  
    model="gpt-3.5-turbo",
    max_tokens=300
    )

    songs1 = song_response1.choices
    if len(songs1) > 0 :
        song_info1 = {
        "Recommended Song": songs1[0].message.content.strip(),
    }
        song_recommendations.append(song_info1)
        songs2 = song_response2.choices
        if len(songs2) > 0 and songs2[0].message.content.strip() != songs1[0].message.content.strip():
            song_info2 = {
            "Recommended Song": songs2[0].message.content.strip(),
        }
            song_recommendations.append(song_info2)
            songs3 = song_response3.choices
            if len(songs3) > 0  and songs3[0].message.content.strip() != songs1[0].message.content.strip() and songs3[0].message.content.strip() != songs2[0].message.content.strip():
                song_info3 = {
                "Recommended Song": songs3[0].message.content.strip(),
            }
                song_recommendations.append(song_info3)

    album_response = client.chat.completions.create(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt_album}
    ],
    model="gpt-3.5-turbo",
    max_tokens=1000,
    )

    album = album_response.choices
    album_recommendation = []
    if len(album) > 0:
        album_info = {
            "This album is for you ": album[0].message.content.strip(),
        }
        album_recommendation.append(album_info)

    return song_recommendations, album_recommendation

# Get user input for mood and language
user_input = st.text_input("Enter the mood and language of songs (e.g., happy/Japanese)")

# Check if the user input matches the expected format
if "/" not in user_input:
    st.warning("Please enter the mood and language in the format 'mood/language'.")
else:
    mood_input, language_input = user_input.split("/")
    # Call the recommend_songs_and_albums function to get song and album recommendations
    song_recommendations, album_recommendation = recommend_songs_and_album(mood_input, language_input)

    st.subheader("Recommended Songs")
    for i in range(len(song_recommendations)):
        song_name = song_recommendations[i].get("Recommended Song", "")
        decorated_song_name = f"{i+1}. {song_name} \n"
        st.write(decorated_song_name)

    # Display the album recommendations
    st.subheader("Album Recommendation")
    album_name = album_recommendation[0].get("This album is for you ", "")
    decorated_album_name = f"{album_name}"

    album_details = album_recommendation[0].get("Detail of the album", "")
    decorated_album_details = f"{album_details}"

    st.write(decorated_album_name)
    st.write(decorated_album_details)