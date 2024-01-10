import os

import streamlit as st
import openai
from pytube import YouTube
import nltk
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd
import sqlite3
from sqlite3 import Error
#Best CS class this semester!!!


# Download the necessary NLTK data
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Function to download YouTube video
def download(link):
    youtube_object = YouTube(link)
    youtube_object = youtube_object.streams.get_highest_resolution()
    try:
        youtube_object.download()
    except:
        st.error("An error has occurred during video download.")
        return None
    return youtube_object.default_filename


# Function to transcribe video audio using OpenAI API
def transcribe_video(audio_path):
    openai.api_key = os.environ.get("OPENAI_API_KEY")

    with open(audio_path, "rb") as audio_file:
        response = openai.Audio.transcribe("whisper-1", audio_file)
        text = response["text"]

    return text

# Function to summarize text using OpenAI API
def summarize_text(text):
    openai.api_key = os.environ.get("OPENAI_API_KEY")

    response = openai.Completion.create(
        engine="davinci-codex",
        prompt=f"Summarize the following text: {text}",
        max_tokens=100,
        temperature=0.3
    )

    summary = response.choices[0].text.strip()

    return summary

# Function to create a POS distribution bar graph
def pos_graph(text):
    # Tokenize and POS tag the text
    tokenized_text = nltk.word_tokenize(text)
    pos_tagged_text = nltk.pos_tag(tokenized_text)

    # Count the occurrences of each POS
    pos_counts = Counter(tag for word, tag in pos_tagged_text)

    # Convert the counter to a DataFrame
    df = pd.DataFrame.from_dict(pos_counts, orient='index').reset_index()
    df.columns = ['Part-of-Speech', 'Count']

    return df

import sqlite3
from sqlite3 import Error

def create_connection():
    conn = None;
    try:
        conn = sqlite3.connect('transcription.db')
        print(f'successful SQLite connection with id {id(conn)}')
    except Error as e:
        print(f'The error {e} occurred')

    return conn

def create_table(conn):
    try:
        sql_create_table = """ CREATE TABLE IF NOT EXISTS transcriptions (
                                        id integer PRIMARY KEY,
                                        title text NOT NULL,
                                        transcription text NOT NULL
                                    ); """
        cursor = conn.cursor()
        cursor.execute(sql_create_table)
    except Error as e:
        print(f'The error {e} occurred')

def insert_transcription(conn, transcription):
    try:
        sql_insert_transcription = '''INSERT INTO transcriptions(title,transcription)
                              VALUES(?,?) '''
        cursor = conn.cursor()
        cursor.execute(sql_insert_transcription, transcription)
        conn.commit()
        return cursor.lastrowid
    except Error as e:
        print(f'The error {e} occurred')


# Streamlit app
def main():
    st.title("YouTube Video Transcription")

    # User input for YouTube link
    youtube_link = st.text_input("Paste the YouTube link here:")

    # Button to download and transcribe the video
    if st.button("Transcribe Video"):
        st.info("Downloading the video...")
        video_filename = download(youtube_link)

        if video_filename:
            st.success("Video downloaded successfully!")
            audio_path = f"./{video_filename}"
            st.info("Transcribing the video audio...")
            transcription = transcribe_video(audio_path)
            st.success("Transcription completed!")

            # Display the transcription text
            st.text_area("Transcription:", transcription)

            # Create and display the POS distribution bar graph
            st.subheader("Part-of-Speech Distribution")
            df = pos_graph(transcription)
            st.bar_chart(df.set_index('Part-of-Speech'))

            # Connect to the DB and insert the transcrip
            conn = create_connection()
            create_table(conn)
            title = YouTube(youtube_link).title
            insert_transcription(conn, (title, transcription))
            conn.close()

# Run the Streamlit app
if __name__ == "__main__":
    main()
