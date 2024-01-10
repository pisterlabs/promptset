# Korzystając z modelu Whisper wykonaj zadanie API (zgodnie z opisem na zadania.aidevs.pl) o nazwie whisper.
# W ramach zadania otrzymasz plik MP3 (15 sekund), który musisz wysłać do transkrypcji,
# a otrzymany z niej tekst odeślij jako rozwiązanie zadania.
from io import BufferedReader
from typing import BinaryIO

import openai
import requests
import random
from helper import Helper


class Whisper:
    @staticmethod
    def generate_answer(test_data):
        # call openai whisper
        openai.api_key = Helper().get_openapi_key()

        # for a Response that point to a file in msg field, go to the url, save the file locally and load it
        if is_not_an_audio_file(test_data):
            url = str(test_data.json()["msg"]).replace("please return transcription of this file: ", "")
            test_data = save_and_load_file_from_url(url)

        transcript = openai.Audio.transcribe("whisper-1", test_data)
        result = transcript["text"]
        test_data.close()

        print(result)
        print(len(result))
        return result


def is_not_an_audio_file(test_data):
    return type(test_data) != BufferedReader


def save_and_load_file_from_url(url):
    file_content = requests.get(url).content.__bytes__()
    random_tmp_nr = random.randint(1, 1000000)
    file = open(f"temp_{random_tmp_nr}.mp3", "wb")
    file.write(file_content)
    file.close()
    return open(file.name, "rb")


if __name__ == '__main__':
    # transcript a file from a disk
    # audio_file = open("./sample_audio.mp3", "rb")
    # Whisper().generate_answer(audio_file)

    # file from a url
    sim_response = save_and_load_file_from_url("https://zadania.aidevs.pl/data/mateusz.mp3")
    Whisper().generate_answer(sim_response)