import os
import errno
import whisper
import json
import openai
from pydub.utils import mediainfo

from audiomind.util import get_env_var


def get_audio_length(audio_file):
    info = mediainfo(audio_file)
    length = float(info["duration"])
    return length


def calculate_transcription_cost(audio_file):
    audio_length = get_audio_length(audio_file)
    cost_per_second = 0.006 / 60
    total_cost = audio_length * cost_per_second
    decimal_places = len(str(cost_per_second).split('.')[1])
    print(f"Length of the audio file: {audio_length:.2f} seconds")
    print(f"Total transcription cost: ${total_cost:.{decimal_places}f}")
    return total_cost


def transcribe_and_store(audio_file, base_name):
    transcript_dir = get_env_var("TRANSCRIPT_DIR")
    os.makedirs(transcript_dir, exist_ok=True)
    transcript_file = f"./{transcript_dir}/{base_name}.txt"
    transcript = transcribe(audio_file, transcript_file)
    try:
        with open(transcript_file, "w") as f:
            if isinstance(transcript, dict):
                json.dump(transcript, f)
            else:
                f.write(transcript)
        return transcript
    except Exception as e:
        print("Error: ", e)
        return transcript


def transcribe(audio_file, transcript_file):
    if get_env_var("FORCE_TRANSCRIPT") != "1":
        if os.path.exists(transcript_file):
            with open(transcript_file, "r") as f:
                transcript = f.read()
                if len(transcript) > 0:
                    print("Transcript found in file. Skipping transcription.")
                    return transcript
    if not os.path.exists(audio_file):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), audio_file)

    if get_env_var("USE_WHISPER_API") == "1":
        return transcribe_openai_api(audio_file)
    result = transcribe_whisper_ondevice(audio_file)
    return result


def transcribe_openai_api(audio_file):
    print("Using OpenAI API to transcribe audio file.")
    calculate_transcription_cost(audio_file)
    audio = open(audio_file, "rb")
    transcript = openai.Audio.transcribe("whisper-1", audio)
    result = transcript["text"]
    return result


def transcribe_whisper_ondevice(audio_file):
    print("Using Local Whisper to transcribe audio file.")
    whisper_model = get_env_var("WHISPER_MODEL")
    model = whisper.load_model(whisper_model, in_memory=True)
    a = model.transcribe(audio_file, verbose=False)
    result = a["text"]
    return result

