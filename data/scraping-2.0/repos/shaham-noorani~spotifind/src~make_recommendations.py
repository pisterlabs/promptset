import openai
import pandas as pd
import json

import os
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")


def read_data():
    liked_songs = pd.read_csv("data/liked_songs.tsv", sep="\t").to_dict("records")
    artist_genres = json.load(open("data/artist_genres.json"))

    return liked_songs, artist_genres


def get_song_analysis(song, artist_genres):
    genres = "/".join(artist_genres[song["Main Artist ID"]])
    return f"{song['Song Title']} by {song['Main Artist']} is a {genres} song released in {song['Release Date']}"


import concurrent.futures


def search_song_batch(song_batch, query, artist_genres):
    matching_songs = []
    search_queries = []

    for song in song_batch:
        song_analysis = get_song_analysis(song, artist_genres)
        search_queries.append(song_analysis)

    search_query = "\n".join(search_queries)

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=[
            {
                "role": "user",
                "content": "I'm going to give you 50 songs and some information about each.",
            },
            {
                "role": "user",
                "content": "I would like for you to tell me if each song strongly matches this query and err on the side of 'no':"
                + query,
            },
            {
                "role": "user",
                "content": "Format your response at {song name} - {yes / no}",
            },
            {
                "role": "user",
                "content": search_query,
            },
        ],
        temperature=0,
        max_tokens=500,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )

    for j, answer in enumerate(response.choices[0].message.content.lower().split("\n")):
        if "yes" in answer:
            matching_songs.append(song_batch[j])

    return matching_songs


def search_songs(query):
    matching_songs = []

    liked_songs, artist_genres = read_data()

    # Create batches of 50 songs
    song_batches = [liked_songs[i : i + 50] for i in range(0, len(liked_songs), 50)]

    # Use ThreadPoolExecutor to make 10 requests of 50 songs at once
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_batch = {
            executor.submit(search_song_batch, batch, query, artist_genres): batch
            for batch in song_batches
        }
        for future in concurrent.futures.as_completed(future_to_batch):
            batch = future_to_batch[future]
            try:
                matching_songs.extend(future.result())
            except Exception as exc:
                print(f"An exception occurred while processing batch {batch}: {exc}")

    return matching_songs
