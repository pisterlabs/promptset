import json
import os
from os.path import join, dirname
from pathlib import Path

import openai
import whisper
from dotenv import load_dotenv
from flask import request
from flask_restx import Resource

from . import api, schema
from .speech_util import SpeechToText

dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)

OPENAPI_KEY = os.getenv('OPENAPI_KEY')
ASSEMBLYAI_KEY = os.getenv('ASSEMBLYAI_KEY')
ROOT_DIR = Path(__file__).parents[3]
AUDIOS_PATH = os.path.join(ROOT_DIR, "audios")

if OPENAPI_KEY:
    openai.api_key = OPENAPI_KEY


def perform_task(query):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=query,
        max_tokens=200
    )
    return response.choices[0].text.strip()

def get_speech_to_text(path):
    model = whisper.load_model("base")

    # load audio and pad/trim it to fit 30 seconds
    audio = whisper.load_audio(path)
    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # detect the spoken language
    _, probs = model.detect_language(mel)
    print(f"Detected language: {max(probs, key=probs.get)}")
    # decode the audio
    options = whisper.DecodingOptions(fp16=False)
    result = whisper.decode(model, mel, options)

    # print the recognized text
    return result.text


def translate_in_desired_language(text, language):
    prompt = f'''
    Your task is to identify the language (in ISO Language Code) the text enclosed in triple back ticks is written in. \
    Then translate that piece of text into the langauge prescribed in <>. \
    The output should be in JSON using 'translation' and 'detected_language' as keys. \
    
    <{language}>
    ```{text}```
    '''
    params = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 60,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0
    }
    # Generate the translation using OpenAI's GPT-3 language model
    response = openai.ChatCompletion.create(**params)

    # Extract the translated text from the response
    output_dict = json.loads(response.choices[0].message.content)
    return output_dict


@api.route("")
class TranslatorOperation(Resource):
    @api.doc("Get Translated to Desired Language")
    @api.marshal_list_with(schema.GetTranslation)
    @api.param("text", required=True)
    @api.param("output_language", required=True)
    def get(self):
        try:
            args = request.args.copy()
            output_lan = args.get("output_language")
            text = args.get("text")
            output_dict = translate_in_desired_language(text, output_lan)
            return {"translation": output_dict['translation'], "detected_language": output_dict['detected_language'],
                    "error": None}, 200
        except Exception as e:
            return {"translation": None, "error": e.__str__()}, 200


@api.route("/speechtotext")
class GetSpeechToText(Resource):
    @api.doc("Speech to Text endpoint")
    @api.marshal_list_with(schema.GetSpeechToText)
    @api.param("audio_path", required=True)
    def get(self):
        try:
            args = request.args.copy()
            audio_path = args.get("audio_path")
            upload_url = SpeechToText().upload_file(ASSEMBLYAI_KEY, audio_path)
            transcript = SpeechToText().create_transcript(ASSEMBLYAI_KEY, upload_url)
            return {"text": transcript['text'], 'language_code': transcript['language_code'], "error": None}, 200
        except Exception as e:
            return {"translation": None, "error": e.__str__()}, 200

    def post(self):
        try:
            file = request.files['file']
            file_path = os.path.join(AUDIOS_PATH, file.filename)
            file.save(file_path)
            text = get_speech_to_text(file_path)
            response = perform_task(text)
            print(response)
            return {'text': response}
        except Exception as e:
            return {"translation": None, "error": e.__str__()}, 200
