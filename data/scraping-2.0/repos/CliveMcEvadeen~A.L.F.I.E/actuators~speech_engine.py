from pathlib import Path
from openai import OpenAI
import os
from pydub import AudioSegment
from pydub.playback import play
from dotenv import load_dotenv
import threading

# load the .env file
load_dotenv()

# get key from .env file
open_ai_api_key = os.getenv("open_ai_api_key")

# define the client
client = OpenAI(api_key=open_ai_api_key)



def play_and_delete(file_path):
    # Load and play the audio file
    audio = AudioSegment.from_mp3(file_path)
    play(audio)

    # Delete the audio file
    os.remove(file_path)


def whisperSpeech(text):
    
    speech_file_path = Path(__file__).parent / "speech.mp3"
    response = client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=text
    )
    response.stream_to_file(speech_file_path)

    # Create a new thread that will execute the play_and_delete function
    thread = threading.Thread(target=play_and_delete, args=(speech_file_path,))

    # Start the new thread
    thread.start()
    
    return 'Completed'
