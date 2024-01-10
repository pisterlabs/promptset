from gtts import gTTS
from time import sleep
from datetime import datetime
import pyglet
import argparse
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


def openai_tts(txt, speech_file_path, voice="echo"):
    if speech_file_path is None:
        speech_file_path = Path(__file__).parent / f"tmp/tts_{voice}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3"
    client = OpenAI()
    response = client.audio.speech.create(model="tts-1", voice=voice, input=txt)

    response.stream_to_file(speech_file_path)
    return speech_file_path


def google_tts(txt):
    tts = gTTS(text=txt, lang='en', tld='co.uk', slow=False)
    filename = 'tmp/temp.mp3'
    tts.save(filename)
    return filename


def say(text, tts="openai"):
    filename = ""
    if tts == "google":
        filename = google_tts(text)
    elif tts == "openai":
        filename = openai_tts(text)
    try:
        music = pyglet.media.load(filename, streaming=True)
        music.play()

        sleep(music.duration)  # prevent from killing
    except Exception as e:
        print(e)
        print("Error playing audio, please try again later")


def main():
    # Create the arguments parser
    default_output_file = f"tmp/tts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3"
    parser = argparse.ArgumentParser(
        description="Takes a markdown file and returns an mp3 file with the tts audio transcription.")
    parser.add_argument(
        'input_file',
        type=str,
        help='The input file to process.')
    parser.add_argument('-o', '--output', type=str,
                        default=default_output_file,
                        help='The output file to write to. Defaults to "tmp/tts_yyyymmdd_hhmmss.mp3".')
    parser.add_argument('-v', '--voice', type=str,
                        default='nova',
                        help='The voice. Valid values: nova, shimmer, echo, onyx, fable, alloy. Defaults to "nova".')

    # Parse the arguments
    args = parser.parse_args()

    with open(args.input_file, 'r') as f:
        text = f.read()
    openai_tts(text, args.output, args.voice)


if __name__ == '__main__':
    main()
