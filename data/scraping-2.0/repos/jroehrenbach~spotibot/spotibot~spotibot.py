
from . import SpotipyWrapper
from . import OpenAIWrapper
import json


class SpotiBot:

    def __init__(self, openai_api_key=None, spotipy_client_id=None, spotipy_client_secret=None,
                 spotipy_redirect_uri=None, spotipy_scope=None):
        self.sp = SpotipyWrapper(spotipy_client_id, spotipy_client_secret,
                                 spotipy_redirect_uri, spotipy_scope)
        self.oa = OpenAIWrapper(openai_api_key)
        self.taste_profile = ""
    
    def initiate_taste_profile(self, user_data, max_tokens=100):
        taste_profile_initiation_messages = [
            {
                "role": "system",
                "content": "You are a music recommendation assistant. Your \
                task is to create a comprehensive music taste profile based \
                on the user's Spotify data. The profile should be detailed \
                and suitable for generating precise music recommendations. \
                It may include the user's preferred genres, favorite artists, \
                and notable liked songs but mainly should summarize the \
                user's music preferences."
            },
            {
                "role": "user",
                "content": f"""
                User data starts here:
                {user_data}
                """
            },
            {
                "role": "user",
                "content": f"Ensure that the music taste profile does not \
                exceed {max_tokens} tokens."
            }
        ]
        self.taste_profile = self.oa.generate_text(
            taste_profile_initiation_messages, max_tokens=max_tokens+20
        )
    
    def update_taste_profile(self, user_data, max_tokens=100):
        taste_profile_update_messages = [
            {
                "role": "system",
                "content": "You are a music recommendation assistant. Your \
                task is to update the user's existing music taste profile \
                based on further user data from Spotify. The music taste \
                profile should be detailed and suitable for generating \
                precise music recommendations. It may include the user's \
                preferred genres, favorite artists, and notable liked songs \
                but mainly should summarize the user's music preferences."
            },
            {
                "role": "user",
                "content": f"""
                Existing music taste profile starts here:
                {self.taste_profile}
                """
            },
            {
                "role": "user",
                "content": f"""
                User data starts here:
                {user_data}
                """
            },
            {
                "role": "user",
                "content": f"Ensure that the music taste profile does not \
                exceed {max_tokens} tokens."
            }
        ]
        self.taste_profile = self.oa.generate_text(
            taste_profile_update_messages, max_tokens=max_tokens+20
        )
    
    def generate_taste_profile(self, user_data, max_tokens=100):
        if self.taste_profile == "":
            self.initiate_taste_profile(user_data, max_tokens=max_tokens)
        else:
            self.update_taste_profile(user_data, max_tokens=max_tokens)
    
    def generate_taste_profile_from_top_artists(self, artists_time_range='medium_term', artists_limit=30, max_tokens=100):
        artists = self.sp.get_current_user_top_artist_names(artists_time_range, artists_limit)
        user_data = "Current top artists" + "\n".join(artists)
        self.generate_taste_profile(user_data, max_tokens=max_tokens)

    def _recommend_tracks_raw(self, user_prompt_input, num_tracks=10, max_tokens=2000):
        recommendation_messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that recommends tracks \
                on Spotify. The recommendations should be based on the user's \
                music taste profile and the specific request mentioned in the \
                prompt. They may have same or similar artists, genres, or \
                other attributes as described in the music taste profile."
            },
            {
                "role": "user",
                "content": f"""
                Music taste profile starts here
                {self.taste_profile}
                """
            },
            {
                "role": "user",
                "content": f"""
                User Prompt: "{user_prompt_input}"
                Number of Tracks Requested: {num_tracks}
                You should only respond with a bulleted list of the tracks in \
                the following format: Track Name - Artist Name
                """
            }
        ]
        return self.oa.generate_text(recommendation_messages, max_tokens=max_tokens)

    def recommend_tracks(self, user_prompt_input, num_tracks=10, max_tokens=2000):
        recommended_tracks_raw = self._recommend_tracks_raw(user_prompt_input, num_tracks, max_tokens)
        processing_messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant transforms a list of \
                tracks into the JSON format"
            },
            {
                "role": "user",
                "content": f"""
                Tracks start here
                {recommended_tracks_raw}
                The JSON in the response should include the "track" and \
                "artist" keys.

                Example:
                [
                    {{
                        "track": "Track Name 1",
                        "artist": "Artist Name 1"
                    }},
                    {{
                        "track": "Track Name 2",
                        "artist": "Artist Name 2"
                    }}
                ]

                Don't write anything else in the response.
                """
            }
        ]
        recommended_tracks_json = self.oa.generate_text(processing_messages, max_tokens=max_tokens)
        # check if the response is valid JSON with "track" and "artist" keys
        try:
            recommended_tracks_json = json.loads(recommended_tracks_json)
            assert all(["track" in track_json and "artist" in track_json for track_json in recommended_tracks_json])
        except:
            recommended_tracks_json = []

        return recommended_tracks_json

    def generate_playlist_name_and_description(self, user_prompt_input, recommended_tracks_json):
        title_messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that creates a title \
                and description for Spotify playlists. The title and \
                description should be based on the users prompt input and \
                the recommended tracks."
            },
            {
                "role": "user",
                "content": f"""
                User Prompt: "{user_prompt_input}"
                Recommended Tracks start here
                {recommended_tracks_json.__str__()}

                Please only write a short and unique title and a description \
                for the playlist. The title should be the first line and the \
                description should be the second line.
                """
            }
        ]
        title_response = self.oa.generate_text(title_messages, max_tokens=400)
        title_results = title_response.split('\n')
        if len(title_results) == 0:
            title_results = ["Untitled", ""]
        if len(title_results) == 1:
            title_results.append('')
        if title_results[0] == "":
            title_results[0] = "Untitled"

        playlist = {
            "name": title_results[0],
            "description": title_results[1]
        }
        return playlist

    def generate_playlist(self, user_prompt_input, num_tracks=10, max_tokens=2000):
        recommended_tracks_json = self.recommend_tracks(user_prompt_input, num_tracks, max_tokens)
        playlist = self.generate_playlist_name_and_description(user_prompt_input, recommended_tracks_json)
        return self.sp.create_playlist_from_json(playlist["name"], playlist["description"], recommended_tracks_json)
