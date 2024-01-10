import openai
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from tqdm import tqdm
import json
import SpotifyGPT


# Change these variables
OPENAI_API_KEY = "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"
CLIENT_ID = "2a407e6bd7524a4c830f7a854f86f516"
CLIENT_SECRET = "dff4b2d7889a48148022b9c31860b815"
REDIRECT_URI = "https://localhost:8888/callback"
USERNAME = '31yhl5xiqievgf7uqxovq2xfbmzu'

PLAYLIST_SOURCE_NAMES = ['Playlist_filter_level_3']
LEGACY_PLAYLIST = "ChatGPT日推Legacy"

output_command = """{
  "songs": [
    {
      "title": "Song1",
      "artist": "Artist1"
    },
    {
      "title": "Song2",
      "artist": "Artist2"
    },
    {
      "title": "Song3",
      "artist": "Artist3"
    }
  ]
}
"""
# query_body = SpotifyGPT(USERNAME, CLIENT_ID, CLIENT_SECRET)

openai.api_key = OPENAI_API_KEY
print("openai version: ", openai.version.VERSION)


# Shows a user's playlists (need to be authenticated via oauth)
username = USERNAME

sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=CLIENT_ID,
                                               client_secret=CLIENT_SECRET,
                                               redirect_uri=REDIRECT_URI,
                                               scope="user-library-read, playlist-modify-private, playlist-read-private, playlist-modify-public user-library-modify"
                                               ))

playlists = sp.user_playlists(username)

my_track_list = ""
count = 0

user_command = ''

for curr_playlist in PLAYLIST_SOURCE_NAMES:
    curr_playlist_id = ''
    for playlist in playlists['items']:
        if curr_playlist == playlist['name']:
            curr_playlist_id = playlist['id']
        if "ChatGPT日推" == playlist['name']:
            user_command = playlist['description']
    items = sp.playlist_items(curr_playlist_id)['items']

    print("Fetching tracks from your playlist:")
    for item in tqdm(items):
        my_track_list += item['track']['name'] + ' by ' + item['track']['artists'][0]['name'] + ', '
        count += 1




print("Fetched", count, "tracks from your playlist")

print("Waiting for recommendations from openai")

preference_result = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Based on my playlist below, what types of music do you think I like?"},
        {"role": "user", "content": my_track_list}
    ]
)
print("================================================================")
print("Your preference analysis from ChatGPT:\n ", preference_result["choices"][0]['message']['content'])
print("================================================================")
print("user_command: ", user_command)


result = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Please recommend 20 music based on my playlist, do not recommend track that is included in my playlist:"},
        {"role": "user", "content": my_track_list},
        {"role": "user", "content": "please also " + user_command + "."},
        {"role": "user", "content": "Answer strictly in the following JSON format: "},
        {"role": "user", "content": output_command + "."}
    ]
)


# def extract_track_artist(string):
#     track_artist_pairs = []
#     tracks_artists = string.split(', ')
#     for track_artist in tracks_artists:
#         parts = track_artist.split(' by ')
#         if len(parts) == 2:
#             track = parts[0].strip()
#             artist = parts[1].strip()
#             track_artist_pairs.append((track, artist))
#     return track_artist_pairs

output = result["choices"][0]['message']['content']
# print(output)
# extracted_list = extract_track_artist(output)
def extract_song_info(json_content):
    # Load JSON content into a Python object
    data = json.loads(json_content)

    # Extract song information
    songs = data["songs"]
    song_dict = {}

    # Iterate over each song and extract title and artist
    for song in songs:
        title = song["title"]
        artist = song["artist"]
        song_dict[title] = artist

    return song_dict

json_start = output.find("{")
json_end = output.rfind("}") + 1
json_content = output[json_start:json_end]

recommend_dict = extract_song_info(json_content)
extracted_list = []
for track in recommend_dict:
    extracted_list.append((track.strip(), recommend_dict[track].strip()))

print("ChatGPT gives", len(extracted_list), "recommendation")

def verify_by_openai(target, result_string):
    result_from_openai = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Do you think these two songs are the same? " + target + " and " + result_string + "?" + "Just answer Yes or No directly without any other wor"},
        ]
    )
    result = result_from_openai["choices"][0]['message']['content']
    if "Yes" in result or "yes" in result:
        return True
    return False

# Search and verify
print("searching and verifying:")
list_to_verify = []
for tup in tqdm(extracted_list):
    query = tup[0] + " " + tup[1]
    result = sp.search(query)
    target = tup[0] + " by " + tup[1]
    result_string = result['tracks']['items'][0]['name'] + " by " + result['tracks']['items'][0]['artists'][0]['name']
    uri = result['tracks']['items'][0]['uri']
    if (verify_by_openai(target, result_string)):
        entry = (uri, target, result_string)
        list_to_verify.append(entry)

print("Successfully found", len(list_to_verify), "tracks on Spotify")

# add to playlist
print("Adding those tracks to your playlist:ChatGPT日推")
playlists = sp.user_playlists(username)
recommend_playlist = ''
legacy_playlist = ''
for playlist in playlists['items']:
    if 'ChatGPT日推' == playlist['name']:
        recommend_playlist = playlist['id']
    if 'ChatGPT日推Legacy' == playlist['name']:
        legacy_playlist = playlist['id']

def get_playlist_id(playlist_name):
    for playlist in playlists['items']:
        if playlist_name == playlist['name']:
            return playlist['id']
    return ''

def get_tracks_from_playlist(playlist_id):
    tracks = set()
    items = sp.playlist_items(playlist_id)['items']
    for item in items:
        tracks.add(item['track']['uri'])
    return tracks
    
def migrate_playlist(playlist_id, legacy_playlist_id):
    tracks_recommend = get_tracks_from_playlist(playlist_id)
    tracks_legacy = get_tracks_from_playlist(legacy_playlist_id)
    for track_uri in tracks_recommend:
        if track_uri not in tracks_legacy:
            sp.playlist_add_items(legacy_playlist_id, [track_uri])
        sp.playlist_remove_all_occurrences_of_items(playlist_id, [track_uri])


tracks_already_exist = set()

recommend_playlist_items = sp.playlist_items(recommend_playlist)['items']
for item in recommend_playlist_items:
    tracks_already_exist.add(item['track']['uri'])

if len(recommend_playlist_items) > 20:
    print("Recommend list has more than 20 ltems, migrate all of them to legacy")
    migrate_playlist(recommend_playlist, legacy_playlist)

print("===============Track Added===============")
added_cout = 0
for entry in list_to_verify:
    if entry[0] not in tracks_already_exist:
        sp.playlist_add_items(recommend_playlist, [entry[0]])
        print(entry[1])
        added_cout += 1
print("===============   Done   ===============")
print(added_cout, "tracks added to your playlist!")