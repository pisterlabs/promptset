from pathlib import Path
from openai import OpenAI
import decouple
from playsound import playsound

api_key = decouple.config('OPENAI_API_KEY')
client = OpenAI(api_key=api_key)

speech_file_path = Path(__file__).parent / "speech.mp3"

def convert_to_text(file_path):
  with open(file_path, 'rb') as file:
      transcript = client.audio.translations.create(
        model='whisper-1',
        file=file
    )
  return transcript

def write_to_file(transcript):
  with open('text.txt', 'w') as file:
      file.write(transcript.text)
  return file


def text_to_speech(file_path):
    
    with open(file_path, "r") as f:
        content = f.read()

    response = client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=f"{content}"
    )
    response.stream_to_file(speech_file_path)
    return speech_file_path

def play_audio(file_path):
    playsound(file_path)

def main(args):
    audio = convert_to_text(args)
    text = write_to_file(audio)
    speech = text_to_speech(text.name)
    play_audio(speech)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert text to speech")
    parser.add_argument("file_path", help="Path to Portuguese audio file")
    args = parser.parse_args()

    main(args.file_path)

