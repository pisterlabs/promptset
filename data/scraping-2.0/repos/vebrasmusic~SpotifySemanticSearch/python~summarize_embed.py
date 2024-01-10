from setup import *
from spotify_handler import *
from genius_handler import *  
import openai 
import pinecone
import re

openai.api_key = openai_api_key

pinecone.init(
    api_key=PINECONE_API_KEY,
    environment=index_environment
)

index_name=index_name

def cleaner(song):
    current_song = song.split('\n')[1:-1] #remove the first and last line to get rid of the song name and artist / embed
    current_song = '\n'.join(current_song)
    return current_song
    
def summarizer(current_song):
    song = cleaner(current_song)
    summary = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", 
            messages=[{"role": "user", "content": 'Write a 3 sentence summary of the following lyrics: ' + song}],
            temperature = 0.6)
    summary = summary.choices[0].message.content
    return summary

def embed_this(summary):
    MODEL = "text-embedding-ada-002"
    res = openai.Embedding.create(
    input=[summary], engine=MODEL)
    embed = [record['embedding'] for record in res['data']]
    return embed

def idGenerator(songName, artist):
    songName = re.sub(r'[^a-z]', '', songName.lower())
    artist = re.sub(r'[^a-z]', '', artist.lower())
    # Create song_id using the first three characters of song_name and the first character of artist_name
    song_id = songName[:3] + artist[0]
    return song_id

def pine(embed, songName, artist, song_id):
    try:
        index = pinecone.Index(index_name)
        to_upsert = [(str(song_id), embed, {"songName": songName, "artist": artist})]
        index.upsert(vectors=to_upsert)
        print(f"Successfully upserted {songName} by {artist} into Pinecone")
    except pinecone.exceptions.PineconeException as e:
        print(f"Error: {e}")



