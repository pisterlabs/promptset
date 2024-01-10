from io import BytesIO
import json
import sys

import openai
from talkotify.microphone import get_audio_from_mic
from .env import OPENAI_API_KEY, init_env, checked_get
from .spotify import get_available_genres, get_device_id, play, search, functions, search_by_genres
from .google import search_by_google


openai.api_key = OPENAI_API_KEY
def voice_to_text() -> str:
    audio = get_audio_from_mic()
    audio_data = BytesIO(audio.get_wav_data())
    audio_data.name = 'from_mic.wav'
    transcript = openai.Audio.transcribe('whisper-1', audio_data, language="ja")
    return transcript['text']

def run():
    question = voice_to_text()
    print(f"user query: {question}")
    device_id = get_device_id()
    print(f"device_id: {device_id}")
    # 1段階目の処理
    # AIが質問に対して使う関数と、その時に必要な引数を決める
    # 特に関数を使う必要がなければ普通に質問に回答する
    messages = [
        {"role": "system", "content": "You are an AI assistant, which search songs and play suitable song for user."},
        {"role": "user", "content": question}
    ]
    while True:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0613",
            messages=messages,
            functions=functions,
            function_call="auto",
        )

        message = response["choices"][0]["message"]
        messages.append(message)
        if message.get("function_call"):
            # 関数を使用すると判断された場合

            # 使うと判断された関数名
            function_name = message["function_call"]["name"]
            # その時の引数dict
            arguments = json.loads(message["function_call"]["arguments"])

            # 2段階目の処理
            # 関数の実行
            if function_name == "play_song":
                print("play: ", arguments["id"])
                play(
                    device_id=device_id,
                    uri=arguments.get("id"),
                )
                break
            elif function_name == "get_available_genres":
                print(f"calling get_available_genres of spotify API")
                function_response = get_available_genres()
                messages.append(
                        {
                            "role": "function",
                            "name": function_name,
                            "content": json.dumps(function_response),
                        },

                )
                continue
            elif function_name == "search_by_genres":
                genres = arguments.get("genres")
                print(f"using spotify genre search: {genres}")
                function_response = search_by_genres(
                    genres=",".join(genres)
                )
                messages.append(
                        {
                            "role": "function",
                            "name": function_name,
                            "content": json.dumps(function_response),
                        },

                )
                continue
            else:
                query = arguments.get("query")
                print(f"using spotify search: {query}")
                function_response = search(
                    query=query
                )
                messages.append(
                        {
                            "role": "function",
                            "name": function_name,
                            "content": json.dumps(function_response),
                        },

                )
                continue
        print("dame")
        print(message["content"])
        break

def raspi_run():
    import RPi.GPIO
    RPi.GPIO.setmode(RPi.GPIO.BCM)
    RPi.GPIO.setup(18, RPi.GPIO.IN)
    print("Press button to talk to play a song")

    while True:
        if RPi.GPIO.input(18) != RPi.GPIO.LOW:
            # HIGH
            try:
                run()
            except Exception as e:
                print(e)
            print("Press button to talk to play a song")


if __name__ == "__main__":
    # run()
    raspi_run()
