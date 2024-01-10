## THIS IS JUST A TEST FILE TO SEE IF I CAN GET THE GPT-3 API TO WORK WITH SPOTIFY PLAYLISTS

import os
import openai
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from dotenv import load_dotenv, find_dotenv

class Config:

    def __init__(self):
        self.debug_mode               = os.environ.get('DEBUG_MODE')
        self.app_key                  = os.environ.get('APP_KEY')
        self.listen                   = os.environ.get('LISTEN')
        self.port                     = os.environ.get('PORT')
        self.spotify_client_id        = os.environ.get('SPOTIPY_CLIENT_ID')
        self.spotify_client_secret    = os.environ.get('SPOTIPY_CLIENT_SECRET')
        self.openai_api_key           = os.environ.get('OPENAI_API_KEY')

        # Tell our app where to get its environment variables from
        dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
        try:
            load_dotenv(dotenv_path)
        except IOError:
            find_dotenv()

class Playlist:
    def __init__(self, playlist_id):
        self.playlist_id = playlist_id

    def get_tracks(self):
        # Authenticate with the Spotify API
        sp_creds = SpotifyClientCredentials()
        sp = spotipy.Spotify(client_credentials_manager=sp_creds)

        # Get the tracks from the playlist
        results     = sp.playlist_items(self.playlist_id)
        tracks      = results['items']
        track_list  = []

        # Append the track name and artist to the list
        for track in tracks:
            track_list.append(track['track']['name'] + ' by ' + track['track']['album']['artists'][0]['name'])

        track_list_str = '\n'.join(track_list)

        return track_list_str

class BlogPost:

    def __init__(self, playlist_id):
        self.playlist_id = playlist_id

    def generate_blog_post(self):
        # Authenticate with the OpenAI API
        cfg = Config()
        openai.api_key = cfg.openai_api_key
        
        play     = Playlist(self.playlist_id)
        playlist = play.get_tracks()

        # Generate a blog post about the playlist
        prompt   = f"My Life in Music is a music blog that posts weekly Spotify playlists every Friday\nYour task is to assume that you are a music blogger who is writing a blog post about the contents of this week's Spotify playlist.\n The playlist is as follows:\n{playlist}"
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {'role': 'system', 'content': 'You are a chatbot'},
                {'role': 'user',   'content': prompt}
            ]
        )
        result = ''
        for choice in response.choices:
            result += choice.message.content

        try:
            return result
        except:
            return result

if __name__ == '__main__':
    playlist_id = '5JjFtIaS0PZM12XBfQhbCz'
    playlist    = Playlist(playlist_id)
    tracks = playlist.get_tracks()
    blogpost    = BlogPost(playlist_id)

    post = blogpost.generate_blog_post()

    print(post)