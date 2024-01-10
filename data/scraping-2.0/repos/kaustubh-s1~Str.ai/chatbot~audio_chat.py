import os
import openai
from dotenv import load_dotenv
from utils.prompt import PROMPT_FOR_GPT

load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY')

class AudioChatApp:
    """
    This is a class for an audio chat app that transcribes user audio input, sends it to OpenAI's
    Whisper model for a response, and returns a transcript of the conversation.
    
    :param audio: The audio parameter is a string representing the filename of an audio file
    """
    def __init__(self):
        self.messages = [
            {"role": "system", "content": PROMPT_FOR_GPT}
        ]

    def transcribe(self, audio):
        audio_filename_with_extension = audio + '.wav'
        os.rename(audio, audio_filename_with_extension)
        
        audio_file = open(audio_filename_with_extension, "rb")
        transcript = openai.Audio.transcribe("whisper-1", audio_file)

        self.messages.append({"role": "user", "content": transcript["text"]})

        response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=self.messages)

        system_message = response["choices"][0]["message"]
        self.messages.append(system_message)

        chat_transcript = ""
        for message in self.messages:
            if message['role'] != 'system':
                chat_transcript += message['role'] + ": " + message['content'] + "\n\n"

        return chat_transcript
