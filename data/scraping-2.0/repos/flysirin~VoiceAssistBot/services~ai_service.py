import openai
from config_data import config
from services import convert_audio

from time import sleep
openai.api_key = config.OPENAI_API_KEY


def transcribe_audio_to_text(file_bytes: bytes = None,
                             file_name: str = 'default_name',
                             path: str = None) -> str:

    path_save_txt = f"temp/{file_name.split('.')[0]}.txt"
    if path and not file_bytes:
        with open(path, "rb") as audio_file_binary:
            file_bytes = audio_file_binary.read()

    def send_request(sound_bytes, name):
        transcript = openai.Audio.transcribe_raw("whisper-1", sound_bytes, f"{name}.mp3")
        with open(path_save_txt, 'w', encoding="utf-8") as f:
            f.write(transcript["text"])
        return path_save_txt

    try:
        return send_request(file_bytes, file_name)

    except openai.error.InvalidRequestError as e:
        if "Invalid file format" in str(e):
            file_bytes = convert_audio.convert_audio_to_mp3(file_bytes=file_bytes, speed=1)
            return send_request(file_bytes, file_name)

    except openai.error.APIConnectionError:
        return send_request(file_bytes, file_name)


def text_request_to_open_ai(text: str = "Say me something good!") -> str:
    def send_request():
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": f"{text}"}])
        return completion.choices[0].message["content"]

    try:
        return send_request()

    except openai.error.APIConnectionError:
        sleep(2)
        return send_request()
