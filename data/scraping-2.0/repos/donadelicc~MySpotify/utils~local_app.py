import spotipy
import openai
import json
import argparse
import datetime
from dotenv import load_dotenv
import os

load_dotenv()
openai.api_key = os.environ.get("OPENAI_API_KEY")


parser = argparse.ArgumentParser(description="Enkelt commandline verktøy")
parser.add_argument("-p", type=str, default="AI genert liste",help="Prompten som beskriver playlisten")
parser.add_argument("-n", type=int, default=10 ,help="Hvor mange sanger ønsker du i playlisten")

args = parser.parse_args()

def get_playlist(prompt, count=10):
    example_json = """ 
    [
      {"song": "Someone Like You", "artist": "Adele"},
      {"song": "Hurt", "artist": "Johnny Cash"},
      {"song": "Fix You", "artist": "Coldplay"},
      {"song": "Nothing Compares 2 U", "artist": "Sinead O'Connor"},
      {"song": "All By Myself", "artist": "Celine Dion"},
      {"song": "Tears in Heaven", "artist": "Eric Clapton"},
      {"song": "My Immortal", "artist": "Evanescence"},
      {"song": "I Can't Make You Love Me", "artist": "Bonnie Raitt"},
      {"song": "Everybody Hurts", "artist": "R.E.M."},
      {"song": "Mad World", "artist": "Gary Jules"}
    ]
    """

    messages = [
        {"role": "system", "content": """You are a helpfull playlist generating assistant.
        You should generate a list of songs and their artists accordning to a text prompt.
        You should retur it as a json array, where each element follows this format: {"song": >song_title>, "artist": <artist_name>}
        """
        },
        {"role": "user", "content": """Generate a playlist of 10 songs based on this prompt: super super sad songs
        """
        },
        {"role": "assistant", "content": example_json
        },
        {"role": "user", "content": f"Generate a playlist of {count} songs based on this prompt: {prompt}"
        },
    ]

    response = openai.ChatCompletion.create(
        messages=messages,
        model="gpt-3.5-turbo",
        max_tokens=400,
    )
    
    playlist = json.loads(response["choices"][0]["message"]["content"])
    return (playlist)

playlist = get_playlist(args.p, args.n)
## JSON format for artists and songs
print(playlist)


sp = spotipy.Spotify(
    auth_manager=spotipy.SpotifyOAuth(
        client_id=os.environ.get("SPOTIFY_CLIENT_ID"),            
        client_secret=os.environ.get("SPOTIFY_CLIENT_SECRET"),
        redirect_uri="http://localhost:8888/",
        scope="playlist-modify-private"
    )
)

current_user = sp.current_user()


track_ids = []
assert current_user is not None


for item in playlist:
        artist, song = item["artist"], item["song"]
        query = f"{song} {artist}"
        search_results = sp.search(q=query, type="track", limit=10)
        track_ids.append(search_results["tracks"]["items"][0]["id"])

playlist_prompt = args.p

created_playlist = sp.user_playlist_create(
    current_user["id"],
    public=False,
    name=f"{'AI - '} {playlist_prompt} {datetime.datetime.now().strftime('%c')}"
    )

sp.user_playlist_add_tracks(
    current_user["id"],
    created_playlist["id"],
    track_ids
    )