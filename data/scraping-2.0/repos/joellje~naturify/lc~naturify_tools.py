from pydantic.v1 import BaseModel, Field
from spotify import start_play_song_by_name, start_play_song_by_lyrics, start_play_song_by_name_and_artist, start_play_song_by_lyrics_and_artist, start_playback, pause_playback, play_next_track, play_previous_track
from lyrics import get_song_lyrics_by_artist_and_song_name_genius, get_song_lyrics_by_song_name_genius, get_song_name_by_lyrics_and_artist_genius, get_song_name_by_lyrics_genius
from langchain.tools import tool

class AccessToken(BaseModel):
    access_token: str = Field(
        description="user's access_token")
class Song(BaseModel):
    access_token: str = Field(
        description="user's access_token")
    song: str = Field(
        description="song name in the user's request")
    
class SongArtist(BaseModel):
    access_token: str = Field(
        description="user's access_token")
    song: str = Field(
        description="song name in the user's request")
    artist: str = Field(
        description="artist in the user's request")
    
class Lyrics(BaseModel):
    access_token: str = Field(
        description="user's access_token")
    lyrics: str = Field(
        description="lyrics in the user's request")
    
class LyricsArtist(BaseModel):
    access_token: str = Field(
        description="user's access_token")
    lyrics: str = Field(
        description="lyrics in the user's request")
    artist: str = Field(
        description="artist in the user's request")
    
@tool("play_song_by_name", return_direct=True, args_schema=Song)
def play_song_by_name_tool(access_token: str, song: str) -> str:
    """Extract the song name from user's request and play the song."""
    return start_play_song_by_name(access_token, song)

@tool("play_song_by_lyrics", return_direct=True, args_schema=Lyrics)
def play_song_by_lyrics_tool(access_token: str, lyrics: str) -> str:
    """Extract the song lyrics from user's request and play the song."""
    return start_play_song_by_lyrics(access_token, lyrics)

@tool("play_song_by_name_and_artist", return_direct=True, args_schema=SongArtist)
def play_song_by_name_and_artist_tool(access_token: str, song: str, artist: str) -> str:
    """Extract the song name and artist from user's request and play the song."""
    return start_play_song_by_name_and_artist(access_token, song, artist)

@tool("play_song_by_lyrics_and_artist", return_direct=True, args_schema=LyricsArtist)
def play_song_by_lyrics_and_artist_tool(access_token: str, lyrics: str, artist: str) -> str:
    """Extract the lyrics and artist from user's request and play the song."""
    return start_play_song_by_lyrics_and_artist(access_token, lyrics, artist)

@tool("get_song_lyrics_by_artist_and_song_name", return_direct=True, args_schema=SongArtist)
def get_song_lyrics_by_artist_and_song_name_tool(access_token: str, artist: str, song: str) -> str:
    """Extract the song name and artist from user's request and get the lyrics."""
    return get_song_lyrics_by_artist_and_song_name_genius(song, artist)

@tool("get_song_lyrics_by_song_name", return_direct=True, args_schema=Song)
def get_song_lyrics_by_song_name_tool(access_token: str, song: str) -> str:
    """Extract the song name from user's request and get the lyrics."""
    return get_song_lyrics_by_song_name_genius(song)

@tool("get_song_name_by_lyrics_and_artist", return_direct=True, args_schema=LyricsArtist)
def get_song_name_by_lyrics_and_artist_tool(access_token: str, artist: str, lyrics: str) -> str:
    """Extract the lyrics and artist from user's request and get the song name."""
    return get_song_name_by_lyrics_and_artist_genius(lyrics, artist)

@tool("get_song_name_by_lyrics", return_direct=True, args_schema=Lyrics)
def get_song_name_by_lyrics_tool(access_token: str, lyrics: str) -> str:
    """Extract the lyrics from user's request and get the song name."""
    return get_song_name_by_lyrics_genius(lyrics)

@tool("start_playback", return_direct=True, args_schema=AccessToken)
def start_playback_tool(access_token: str) -> str:
    """Starts or resumes user's playback."""
    return start_playback(access_token)

@tool("pause_playback", return_direct=True, args_schema=AccessToken)
def pause_playback_tool(access_token: str) -> str:
    """Pauses user's playback."""
    return pause_playback(access_token)

@tool("play_next_track", return_direct=True, args_schema=AccessToken)
def play_next_track_tool(access_token: str) -> str:
    """Skips the current track and plays the next track."""
    return play_next_track(access_token)

@tool("play_previous_track", return_direct=True, args_schema=AccessToken)
def play_previous_track_tool(access_token: str, lyrics: str) -> str:
    """Plays the previous track."""
    return play_previous_track(access_token)



music_player_tools = [
    play_song_by_name_tool,
    play_song_by_lyrics_tool,
    play_song_by_name_and_artist_tool,
    play_song_by_lyrics_and_artist_tool,
    get_song_lyrics_by_artist_and_song_name_tool,
    get_song_name_by_lyrics_and_artist_tool,
    get_song_lyrics_by_song_name_tool,
    get_song_name_by_lyrics_tool,
    start_playback_tool,
    pause_playback_tool,
    play_next_track_tool,
    play_previous_track_tool
]