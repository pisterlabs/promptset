import os
import sys
from openai import OpenAI
from pydub import AudioSegment

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
assert OPENAI_API_KEY, ("OPENAI_API_KEY env variable not set")
client = OpenAI(api_key=OPENAI_API_KEY)

def transcribe_audio(api_key, audio_file):
    return client.audio.transcribe("whisper-1", audio_file)["text"] # type: ignore

def translate(api_key, audio_file):
    return client.audio.translate("whisper-1", audio_file)["text"]  # type: ignore

def get_audio_file(path):
    # read a a file frop path with rb mode
    return open(path, "rb")

def save_transcription(transcription, output_path, audio_file_path):
    base_file_name = os.path.splitext(os.path.basename(audio_file_path))[0]
    transcription_file_path = os.path.join(output_path, f"{base_file_name}_text")
    with open(transcription_file_path, "w") as f:
        f.write(transcription)


def convert_audio(input_file, output_format):
    # convert audio file to output_format format supporting mp3, mp4, mpeg, mpga, m4a, wav, or webm (opus)
    # input file format is detected automatically by input file extension
    # https://www.freeconvert.com/opus-to-mp3/download
    input_format = input_file.split(".")[-1]
    if input_format == output_format:
        print("Debug: nothing to convert")
        return input_file
    elif input_format in ["mp3", "mp4", "mpeg", "mpga", "m4a", "wav", "webm"]:
        print("Debug: nothing to convert")
        return input_file
    audio = AudioSegment.from_file(input_file, format=input_format)
    output_file = input_file.replace(input_format, output_format)
    audio.export(output_file, format=output_format)
    return output_file

def translate_audio(file_name):
    # when already english just does speech to text
    audio_folder_path = os.getcwd()
    audio_file_path = os.path.join(audio_folder_path, file_name)
    output_path = os.getcwd()
    FORMAT = "mp3"
    formated_audio = convert_audio(audio_file_path, FORMAT)
    audio_file = get_audio_file(formated_audio)
    transcription = translate(OPENAI_API_KEY, audio_file)
    audio_file.close()
    print(transcription)
    save_transcription(transcription, output_path, audio_file_path)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 translate.py <input_file>")
        sys.exit(1)
    
    translate_audio(sys.argv[1])
