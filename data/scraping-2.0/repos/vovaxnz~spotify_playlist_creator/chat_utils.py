from dataclasses import dataclass
from datetime import datetime
import json
import os
import time
import openai
from typing import Dict, List

openai.api_key = os.getenv("OPENAI_API_KEY")

analyse_request_prompt = "I need to pick a playlist of songs. Here are my requirements: {text}.\nTake a deep breath and analyze my request, think about what genres, artists, bands which can be appropriate. Don't give a direct list of songs. Instead, write your thoughts and  analysis of my request (2-3 sentences). Tracks i want should be not very trivial and not popular"

songs_in_json_prompt = "According to my request and the information you provided, create a list of 20 songs in JSON format. Give the result in this format:/n/n[{\"artist\": \"...\", \"song\": \"...\"}, ...].\n\nDon't write anything other than JSON. Start your message with \"[\""

describe_playlist_prompt = "In two sentences, describe the playlist you've picked up."

create_title_prompt = "Summarize music i requested in 1-3 words. Write without quotes."


@dataclass
class Song:
    title: str
    artist: str


class Playlist: # TODO: Move
    def __init__(self, title: str, description: str, songs: List[Song]):

        curr_date_str = datetime.now().strftime("%Y.%m.%d") #  %H-%M-%S
        self.title = f"{curr_date_str}: {title}"
        self.description = description
        self.songs = songs

    def remove_songs(songs: List[Song]):
        ... # TODO


class PlaylistGeneratorError(Exception):
    """Raised when any error occurs in the PlaylistGenerator"""
    pass


class PlaylistGenerator:

    def __init__(self) -> None:
        self._conversation = list()

    def recieve_request(self, text) -> str:
        self._analyse_request(text)
        return self._conversation[-1]["content"]

    def generate_playlist(self) -> Playlist:
        songs = self.get_playlist_content()
        description = self.get_playlist_description()
        title = self.get_playlist_title()
        self._conversation = list()
        return Playlist(title=title, description=description, songs=songs)

    def _analyse_request(self, text: str) -> str:
        input_text = analyse_request_prompt.format(text=text)
        output = self._speak_to_model(input_text)
        self._save_model_answer(output)
                    
    def get_playlist_content(self) -> List[Song]:
        for i in range(5):
            output = self._speak_to_model(songs_in_json_prompt)
            result = list()
            try:
                songs = json.loads(output)
                for song in songs:
                    result.append(Song(title=song["song"], artist=song["artist"]))
                break
            except:
                time.sleep(1)
        if len(result) == 0:
            raise PlaylistGeneratorError("Can not obtain a songs list") 
        self._save_model_answer(output)
        return result
    
    def get_playlist_description(self) -> str:
        output = self._speak_to_model(describe_playlist_prompt, top_last=3)
        self._save_model_answer(output)
        return output
    
    def get_playlist_title(self) -> str:
        output = self._speak_to_model(create_title_prompt, top_last=3)
        self._save_model_answer(output[:100])
        return output

    def _speak_to_model(self, text: str, top_last: int = None) -> str: 
        print("You: ", text)
        self._conversation.append({"role": "user", "content": text})
        if top_last is not None:
            input_conversation = self._conversation[-top_last:]
        else:
            input_conversation = self._conversation
        output = self._get_api_response(messages=input_conversation)
        print("Model: ", output)
        return output
    
    def _save_model_answer(self, text: str):
        self._conversation.append({"role": "assistant", "content": text})

    def _get_api_response(self, messages: List[Dict]) -> str:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.5,
        )
        return response["choices"][0]["message"]["content"]