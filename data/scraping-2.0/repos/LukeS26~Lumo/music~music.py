import threading, signal
import subprocess
import json
import psutil
import random
import re
import openai
import os
import time
import whisper_timestamped as whisper

from config.config_variables import api_credentials

openai.api_key = api_credentials["openai"]["key"]

class MusicController:
    def __init__(self):
        # The psutil process running ffplay
        self.music_player = None
        
        self.player_lock = False

        # The current song being played, as its id
        self.current_song = -1
       
        # The current lyric of the song playing
        self.lyric_index = -1

        self.paused = False
        
        # Variables for playback time calculation
        self.pause_time = 0
        self.paused_at = 0
        self.seek_start = 0

        # The dicts of artists, albums, and songs
        self.artists = json.load(open("./music/artists.json"))
        self.albums = json.load(open("./music/albums.json"))
        self.songs = json.load(open("./music/songs.json"))

        self.used_songs = []
        self.played_list = []
        self.available_songs = []

        threading.Thread(target=self.music_loop, name="music_loop").start()
        threading.Thread(target=self.lyric_loop, name="lyric_loop").start()
        
        signal.signal(signal.SIGINT, signal.SIG_DFL)

        while self.music_player is None:
            pass
        self.pause()

    def set_music_volume(self, volume):
        # This is way more complicated then it has any right being
        pass

    # Gets the current playback time through the song. 
    # To do so it takes the current time since the creation of the music player, and subtracts the amount of time it has been paused
    # If the music is paused, then it uses the time it was paused at, instead of the current time, since in that case it won't be playing
    def get_play_time(self):
        if self.music_player is None:
            return 0

        if self.is_paused():
            return (self.paused_at - self.music_player.create_time()) - self.pause_time + self.seek_start
        else:
            return (time.time() - self.music_player.create_time()) - self.pause_time + self.seek_start

    # Method to check if the music_process isn't running. If the music process doesn't exist it also returns True
    def is_paused(self):
        return self.paused

    # Shortcut to return the id of the currently playing song
    def get_current_song(self):
        return self.played_list[self.current_song]

    # Suspends the process, and tracks the time at which it was paused for calculating play time
    def pause(self):
        self.music_player.suspend()
        self.paused_at = time.time()
        self.paused = True
                
    # Resumes the suspended process, and updates the total amount of time it was paused to subtract from the play time as needed
    def unpause(self):        
        self.music_player.resume()
        self.pause_time += time.time() - self.paused_at
        self.paused = False

    # Skips forwards or backwards by some number of songs
    # To do this it first obtains a lock on the player, to prevent the music loop from starting the next song as soon as it kills the process
    # It then gets the next song, num forward or back, and plays it
    def skip_songs(self, num):
        if self.player_lock:
            return

        self.player_lock = True
        
        self.current_song = self.get_next_song(num)
        
        self.play()

    # Seeks forwards or backwards by some number of seconds
    # To do this it first obtains a lock on the player, to prevent the music loop from starting the next song as soon as it kills the process
    # It then calculates the point at which to seek, clamping it between 0 and the song duration, and plays it
    def seek(self, seconds):
        if self.player_lock:
            return

        self.player_lock = True
    
        seek_time = min(max(0, self.get_play_time() + seconds), float(self.songs[self.get_current_song()]["duration"]))

        self.play(seek_time=seek_time)

    # Begin the music_player process
    # To begin the process, it resets kills the current music_player if it's running, and resets any variable needed
    # It then creates the arg list, with a -ss starting point if it has a seek time provided
    # It then starts the process, saving it to the music_player variable
    def play(self, seek_time=0):
        if not (self.music_player is None) and self.music_player.is_running():
            self.music_player.kill()
        
        self.pause_time = 0
        self.seek_start = seek_time
        self.lyric_index = -1
        self.paused = False

        if seek_time > 0:
            args = ["ffplay", "-autoexit", "-nodisp", "-ss", str(self.seek_start), self.songs[self.get_current_song()]["link"]]
        else:
            args = ["ffplay", "-autoexit", "-nodisp", self.songs[self.get_current_song()]["link"]]
        
        self.music_player = psutil.Process(subprocess.Popen(
            args=args,
            stdout=subprocess.PIPE,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE
        ).pid)

        print(f"Now playing {self.songs[self.get_current_song()]['name']} by {self.songs[self.get_current_song()]['artist']}" )

    # Method to calculate the song to play, based on how many forward or backwards it wants to go.
    # if the new index would be above the length of the play list, it adds new songs to that list, and sets itself to the last index
    # If it would be negative it becomes 0 instead
    def get_next_song(self, num):
        new_song_index = self.current_song + num
        if new_song_index >= len(self.played_list):
            num_to_add = new_song_index - len(self.played_list) + 1
            for i in range(num_to_add):
                self.pick_new_song()

            new_song_index = len(self.played_list) - 1

        if new_song_index < 0:
            new_song_index = 0

        return new_song_index

    def pick_new_song(self):    
        picked_song = 0
        if self.shuffle:
            picked_song = random.randint(0, len(self.available_songs) - 1)
        
        self.current_song += 1

        self.used_songs.append(self.available_songs[picked_song])
        self.played_list.append(self.available_songs[picked_song])
        self.available_songs.pop(picked_song)

        if len(self.played_list) > (len(self.available_songs) + len(self.used_songs)) :
            self.played_list.pop(0)
            self.current_song -= 1

        if self.loop and len(self.used_songs) > 0.2 * (len(self.available_songs) + len(self.used_songs)):
            self.available_songs.append(self.used_songs.pop(0))

    def music_loop(self, shuffle=True, loop=True, album=None, artist=None, playlist=None):
        self.shuffle = shuffle
        self.loop = loop

        if album:
            try:
                self.available_songs = self.albums[album]["songs"]
            except:
                print(f"Album {album} not found!")
        elif artist:
            try:
                self.available_songs = self.artists[artist]["songs"]
            except:
                print(f"Artist {artist} not found!")
        elif playlist:
            pass
        else:
            self.available_songs = list(self.songs.keys())

        while True:
            if not self.player_lock:
                self.current_song = self.get_next_song(1)
                self.play()
            else:
                self.player_lock = not self.music_player.is_running()

            if not self.player_lock:
                self.music_player.wait()

    def get_current_lyric (self):
        current_song = self.get_current_song()
        if "lyrics" not in self.songs[current_song].keys():
            return (-1, "")

        cur_play_time = self.get_play_time()
        possible_lyrics = self.songs[current_song]["lyrics"]

        for i, (_, lyric_time) in enumerate(possible_lyrics):
            if cur_play_time < lyric_time:
                if i > 0:
                    return (i - 1, possible_lyrics[i - 1][0])
                else:
                    return (-1, "")
        
        if possible_lyrics:
            return len(possible_lyrics) - 1, possible_lyrics[-1][0]

        # No lyrics found
        return (-1, "")
        

    def lyric_loop(self):
        while True:
            time.sleep(0.1)

            index, lyric = self.get_current_lyric()

            if index > self.lyric_index:
                print(lyric)
                self.lyric_index = index

    def control_music(self, command):
        if  "pause" in command or "stop" in command:
            self.pause()()
            return f"Music paused. Current song: {self.songs[self.get_current_song()]['name']}"
        elif "unpause" in command or "resume" in command or "play" in command:
            self.unpause()
            return f"Music resumed. Current song: {self.songs[self.get_current_song()]['name']}"
        elif "rewind" in command or "restart" in command:
            self.skip_songs(0)
            return f"Song rewound. Current song: {self.songs[self.get_current_song()]['name']}"
        elif "back" in command or "previous" in command:
            num = -1
            for word in command:
                if word.isdigit():
                    num = -1 * int(word)
            self.skip_songs(num)
            return f"Went back {num} songs. Current song: {self.songs[self.get_current_song()]['name']}"
        elif "next" in command or "forward" in command or "skip" in command:
            num = 1
            for word in command:
                if word.isdigit():
                    num = int(word)
            self.skip_songs(num)
            return f"Skipped ahead {num} songs. Current song: {self.songs[self.get_current_song()]['name']}"

        return f"Unknown command: {command}"

class MusicSetup:
    def __init__(self):
        pass

    def initialize_all_music(self, do_lyrics):
        duration_re = re.compile(r"\d+\.\d+")

        artists = {}
        albums = {}
        songs = {}

        old_songs = json.load(open("./music/songs.json"))

        artist_list = next(os.walk("./music/music_library"))[1]
        
        for artist in artist_list:
            tags = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "system", "content": f"Create a list of tags for the band {artist}, seperated by ', '"}])

            similar_bands = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[
                {"role": "system", "content": f"{artist_list}"},
                {"role": "system", "content": f"Create a list of bands from the provided list that are similar to {artist}, seperated by ', '"}    
            ])

            artists[artist] = {
                "name": artist,
                "albums": [],
                "songs": [],
                "tags": tags.choices[0].message.content.split(", "),
                "similar_to": similar_bands.choices[0].message.content.split(", ")
            }

            album_list = next(os.walk(f"./music/music_library/{artist}"))[1]
            for album in album_list:
                artists[artist]["albums"].append(album)

                albums[album] = {
                    "name": album,
                    "artist": artist,
                    "songs": []
                }
                
                album_hash = str(hash(album))
                
                song_list = next(os.walk(f"./music/music_library/{artist}/{album}"))[2]
                for song in song_list:
                    song_name = song.replace(".mp3", album_hash)
                    song_link = f"./music/music_library/{artist}/{album}/{song}"

                    args = ("ffprobe","-show_entries", "format=duration", "-i", song_link)
                    popen = subprocess.Popen(args, stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
                    popen.wait()
                    output = popen.stdout.read()

                    artists[artist]["songs"].append(song_name)
                    albums[album]["songs"].append(song_name)

                    songs[song_name] = {
                        "name": song.replace(".mp3", ""),
                        "album": album,
                        "artist": artist,
                        "link": song_link,
                        "duration": duration_re.search(str(output)).group(0)
                    }

                    if song_name in old_songs.keys():
                        songs[song_name]["lyrics"] = old_songs[song_name]["lyrics"]

        with open("music/artists.json", 'w') as fp:
            json.dump(artists, fp)

        with open("music/albums.json", 'w') as fp:
            json.dump(albums, fp)

        with open("music/songs.json", 'w') as fp:
            json.dump(songs, fp)

        if do_lyrics:
            self.add_lyrics_to_songs()

    def add_lyrics_to_songs(self):
        with open("music/songs.json", 'rb') as fp:
            songs = json.load(fp)

            for (id, song_data) in songs.items():
                if not "lyrics" in song_data.keys(): 
                    print(id)
                    songs[id]["lyrics"] = self.generate_song_lyrics(song_data["name"], song_data["link"])

                    with open("music/songs.json", 'w') as f:
                        json.dump(songs, f)


    def generate_song_lyrics(self, song_name, file_link):
        audio = whisper.load_audio(file_link)

        model = whisper.load_model("large-v2", device="cpu")

        result = whisper.transcribe(model, audio, language="en", initial_prompt=f"Transcribe the song lyrics for {song_name}")

        lines = result["segments"]
        
        lyrics = []

        for line in lines:
            lyrics.append((line["text"], line["start"]))
        
        return lyrics

import spotipy
from spotipy.oauth2 import SpotifyOAuth
import webbrowser

scopes = [
    "user-read-playback-state",
    "user-modify-playback-state",
    "user-read-currently-playing",
    "app-remote-control",
    "playlist-read-private",
    "playlist-read-collaborative",
    "playlist-modify-private",
    "playlist-modify-public",
    "user-follow-modify",
    "user-follow-read",
    "user-read-playback-position",
    "user-top-read",
    "user-read-recently-played",
    "user-library-modify",
    "user-library-read",
    "user-read-email",
    "user-read-private",
    "streaming"
]


class SpotifyController:
    def __init__(self):
        self.spotify = spotipy.Spotify(auth_manager=SpotifyOAuth(
            client_id=api_credentials["spotify"]["client_id"],
            client_secret=api_credentials["spotify"]["client_secret"],
            redirect_uri="http://localhost:8002",
            scope=scopes
        ))

        self.current_song = self.spotify.currently_playing()

    def get_access_token(self, refresh_token=None):
        if refresh_token is None:
            token = self.spotify.auth_manager.get_access_token()
        else:
            token = self.spotify.auth_manager.refresh_access_token(refresh_token)

        return (token["access_token"], token["refresh_token"])

    def get_devices(self):
        return self.spotify.devices()

    def get_playlists(self):
        user = self.spotify.current_user()

        playlists = self.spotify.user_playlists(user=user["id"])["items"]

        simple_playlists = {}

        for playlist in playlists:
            simple_playlists[playlist["name"]] = playlist["id"]

        return simple_playlists

    def pause(self):
        devices = self.spotify.devices()["devices"]

        self.spotify.pause_playback(device_id=devices[0]["id"])

    def unpause(self):
        devices = self.spotify.devices()["devices"]

        self.spotify.start_playback(device_id=devices[0]["id"])

    def skip_songs(self, num):
        if num < 0:
            for i in range(num):
                self.spotify.previous_track()
        elif num > 0:
            for i in range(num):
                self.spotify.next_track()
        else:
            self.spotify.seek_track(0)

    def get_play_time(self):
        song = self.spotify.currently_playing()
        if not song is None:
            self.current_song = song

        if not self.current_song is None:
            return self.current_song["progress_ms"] / 1000   
