from flask import request, Blueprint, Flask, make_response, jsonify

import os
import json

from typing import Any

import openai
from pydub import AudioSegment
from pydub.silence import detect_nonsilent


app = Flask(__name__)
app_routes = Blueprint("app_routes", __name__)
app.register_blueprint(app_routes)


HOMEDIR = os.path.expanduser("~")
APPDATA_PATH = f"{HOMEDIR}/structured-voice-logging/dev_app_data"
LOGFILES_DIR = f"{APPDATA_PATH}/logfiles"


def remove_silence(audio_file, silence_threshold=-20, min_silence_duration=100, padding=50):
    audio = AudioSegment.from_wav(audio_file)
    nonsilent_ranges = detect_nonsilent(audio, min_silence_len=min_silence_duration, silence_thresh=silence_threshold)

    if not nonsilent_ranges:
        return None

    trimmed_audio = AudioSegment.empty()
    for start, end in nonsilent_ranges:
        start = max(0, start - padding)
        end += padding
        trimmed_audio += audio[start:end]

    return trimmed_audio


def _get_write_type(content: Any):
    if isinstance(content, str):
        return 'w'
    elif isinstance(content, bytes):
        return 'wb'
    else:
        raise TypeError(f"content must be str or bytes, not {type(content)}")
        
        
class FileSystem:
    def __init__(self, root: str) -> None:
        self.root = root
    
    def save(self, path: str, content: str) -> None:
        with open(os.path.join(self.root, path), _get_write_type(content)) as f:
            f.write(content)


class WhisperTranscriber:
    async def transcribe(self, file):
        with open(file, 'rb') as audio:
            transcript = openai.Audio.transcribe("whisper-1", audio)
        print(json.dumps(transcript, indent=4))
        return transcript


def remove_silence_and_save(input_file, output_file) -> bool:
    trimmed_audio = remove_silence(input_file)
    if trimmed_audio:
        trimmed_audio.export(output_file, format="wav")
        return True
    else:
        return False


def delete_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
    else:
        print(f"The file {file_path} does not exist")
    

filesystem = FileSystem(root=APPDATA_PATH)
transcriber = WhisperTranscriber()

@app_routes.route("/transcribe", methods=["POST"])
async def transcribe():
    print("Entering routes.transcribe...")
    
    audio_data = request.get_data()
    user_id = request.args.get('userId', None)
    print(f"userId: {user_id}")

    print("/transcribe: len(audio_data):", len(audio_data))
    
    dest_dir = os.path.join(filesystem.root, user_id, "recordings")
    os.makedirs(dest_dir, exist_ok=True)
    
    destpath = f"{dest_dir}/rec1.wav"
    app.logger.info(f"Writing to '{destpath}'.")
    with open(destpath, "wb") as f:
        f.write(audio_data)
    app.logger.info("Done writing.")

    trimmed_path = destpath.replace(".wav", "_trimmed.wav")
    has_audio = remove_silence_and_save(destpath, trimmed_path)
    
    if has_audio:
        app.logger.info("Transcribing...")
        transcript = await transcriber.transcribe(destpath)
        app.logger.info("Done transcribing.")
        print(transcript)
    else:
        app.logger.info("No non-silent audio detected.")
        transcript = "SYSTEM: No audio detected. Output an empty list."
    response_data = {'transcription': transcript}
    
    delete_file(destpath)
    delete_file(trimmed_path)

    return make_response(jsonify(response_data))
