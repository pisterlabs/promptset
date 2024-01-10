import openai
from dotenv import load_dotenv
import os
from pydub import AudioSegment

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

filename = 'data/2830-3980-0043.wav'

# audio_file= open(filename, "rb")
# transcript = openai.Audio.transcribe("whisper-1", audio_file)
# print(transcript)
# audio_file.close()


### Transcribe
def speech_to_text(filename):
    """
    Transcribe audio file into text
    """
    
    try:
        # sound = AudioSegment.from_file(filename)
        # sound.export(filename, format="mp3")
        with open(filename, 'rb') as audio_file:
            transcript = openai.Audio.transcribe(
            file = audio_file,
            model = "whisper-1",
            response_format="text",
            language="en"
        )
        return transcript
    except Exception as e:
        print(e)
        transcript = "Error"

        return transcript


#### Subtitles        
def speech_to_sub(filename):
    """
    Transcribe audio file into subtitles
    """
    try:
        with open(filename, 'rb') as audio_file:
            subtitles = openai.Audio.transcribe(
            file = audio_file,
            model = "whisper-1",
            response_format="srt",
            language="en"
        )
        return subtitles
    except Exception as e:
        print(e)
        subtitles = "Error"

        return subtitles


### Prompt modification
def speech_prompt_modification(filename):
    """
    Optimize outputs with prompts
    """
    try:
        with open(filename, 'rb') as audio_file:
            prompt_modification = openai.Audio.transcribe(
            file = audio_file,
            model = "whisper-1",
            response_format="text",
            language="en",
            prompt="In surprice"
        )
        return prompt_modification
    except Exception as e:
        print(e)
        prompt_modification = "Error"

        return prompt_modification


### Translation
def speech_translation(filename):
    """
    Translate spanish audio file into english text
    """
    try:
        with open(filename, 'rb') as audio_file:
            translation = openai.Audio.translate(
            file = audio_file,
            model = "whisper-1",
            response_format="text",
            language="en"
            )
        return translation
    except Exception as e:
        print(e)
        translation = "Error"

        return translation

print(speech_to_text(filename))

print(speech_to_sub(filename))

print(speech_prompt_modification(filename))