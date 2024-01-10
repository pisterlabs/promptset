import spotipy
from spotipy.oauth2 import SpotifyOAuth
import openai
from src.config import settings

scope = "user-library-read,user-top-read"

sp = spotipy.Spotify(
    auth_manager=SpotifyOAuth(
        scope=scope,
        client_id=settings.spotipy_client_id,
        client_secret=settings.spotipy_client_secret,
        redirect_uri="http://localhost:8080/callback",
    )
)

sp._auth_manager.get_access_token(as_dict=False)
print("Done")
