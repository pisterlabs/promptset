import tempfile
import urllib.request

import openai
import telebot

from bot.constants import bot

from pydub import AudioSegment

from db.db import insert_voice, get_api_key
from open_ai.common import GET_FILE_URL
from settings import TEMP_OGG_DIR, TEMP_MP3_DIR


def transcribe_voice(message: telebot.types.Message):
    api_key = get_api_key(user_id=message.from_user.id, key_type='openai')
    path_array = download_voice_as_mp3(voice=message.voice)
    with open(path_array[1].name, 'rb') as mp3_file:
        transcribed_voice_response = openai.Audio.transcribe(
            model='whisper-1',
            file=mp3_file,
            api_key=api_key
        )
    transcribed_voice = transcribed_voice_response.get('text')
    insert_voice(message=message, path=path_array[0], transcribed_voice=transcribed_voice)
    path_array[1].close()
    return transcribed_voice


def translate_voice(message: telebot.types.Message):
    api_key = get_api_key(user_id=message.from_user.id, key_type='openai')
    path_array = download_voice_as_mp3(voice=message.voice)
    with open(path_array[0], 'rb') as mp3_file:
        translated_voice_response = openai.Audio.translate(
            model='whisper-1',
            file=mp3_file,
            api_key=api_key
        )
    translated_voice = translated_voice_response.get('text')
    insert_voice(message=message, path=path_array[1], translated_voice=translated_voice)
    path_array[1].close()
    return translated_voice


def download_voice_as_mp3(voice: telebot.types.Voice):
    voice_file_path = bot.get_file(voice.file_id).file_path
    tg_file_url = GET_FILE_URL(path=voice_file_path)
    tmp_ogg_file = tempfile.NamedTemporaryFile(dir=f'{TEMP_OGG_DIR}', suffix='.ogg')
    tmp_mp3_file = tempfile.NamedTemporaryFile(dir=f'{TEMP_MP3_DIR}', suffix='.mp3')
    audio_file = urllib.request.urlopen(tg_file_url)
    with open(tmp_ogg_file.name, 'wb') as tmp_file:
        tmp_file.write(audio_file.read())
    AudioSegment.from_ogg(tmp_ogg_file.name).export(
        out_f=tmp_mp3_file.name,
        format='mp3'
    )
    tmp_ogg_file.close()
    return [voice_file_path, tmp_mp3_file]
