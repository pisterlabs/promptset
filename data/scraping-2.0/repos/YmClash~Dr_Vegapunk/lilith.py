import spotipy.util as util
import os ,yaml
from dotenv import load_dotenv
from spotipy.oauth2 import SpotifyClientCredentials,SpotifyOAuth
from langchain.agents.agent_toolkits.openapi.spec import reduce_openapi_spec

from langchain.requests import RequestsWrapper


load_dotenv()



#
with open("spotify_openapi.yaml") as f:
    raw_spotify_api_spec = yaml.load(f, Loader=yaml.Loader)
spotify_api_spec = reduce_openapi_spec(raw_spotify_api_spec)

SPOTIPY_CLIENT_ID = os.getenv('SPOTIPY_CLIENT_ID')
SPOTIPY_CLIENT_SECRET = os.getenv('SPOTIPY_CLIENT_SECRET')
# SPOTIPY_REDIRECT_URI = os.getenv('SPOTIPY_REDIRECT_URI')

def construct_spotify_auth_headers(raw_spec: dict):
    scopes = list(raw_spec['components']['securitySchemes']['oauth_2_0']['flows']['authorizationCode']['scopes'].keys())
    auth_manager = SpotifyOAuth(client_id=SPOTIPY_CLIENT_ID, client_secret=SPOTIPY_CLIENT_SECRET,scope=scopes)

    return {
        'Authorization': f'Bearer {auth_manager}'
    }


headers = construct_spotify_auth_headers(raw_spotify_api_spec)
request_wrapper = RequestsWrapper(headers=headers)
