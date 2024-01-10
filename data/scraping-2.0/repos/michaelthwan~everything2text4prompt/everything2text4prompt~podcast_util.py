import re

import openai
import requests

from .util import chunk_mp3, PodcastData

class PodcastUtil:
    @staticmethod
    def convert_podcast_transcript(podcast_url):
        def download_mp3(url: str, file_path: str):
            with open(file_path, "wb") as file:
                response = requests.get(url)
                file.write(response.content)

        content = requests.get(podcast_url)
        mp3_url = re.findall("(?P<url>\;https?://[^\s]+)", content.text)[0].split(';')[1]
        print(f"mp3_url: {mp3_url}")
        mp3_file_path = "temp.mp3"
        download_mp3(mp3_url, mp3_file_path)
        print(f"Downloaded mp3 file")
        file_part_list = chunk_mp3(mp3_file_path)
        transcript_list = []
        for file_part in file_part_list:
            file = open(file_part, "rb")
            print(f"Calling openai whisper-1 for {file_part}")
            transcript = openai.Audio.transcribe("whisper-1", file)
            transcript_list.append(transcript)
        print(transcript_list)
        title = description = ""  # TODO
        return PodcastData(" ".join(transcript_list), title, description), True, "Success"
