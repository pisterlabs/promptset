import openai
from gtts import gTTS
from playsound import playsound
import config
from datetime import datetime


class Utilies():
    def __init__(self) -> None:
        pass

    def datetime(self, mode):
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        months = ['','January','February','March','April','May','June','July','August','September','October','November','December']
        date, time = now.split(' ')
        year, month, day = date.split('-')
        hour, mins, sec = time.split(':')

        prompt = ['PM' if int(hour)>=12 else 'AM']
        hour = [int(hour)%12 if int(hour)%12!=0 else 12]

        today = f"Today's date is {day} of {months[int(month)]} {year}"
        dTime = f"The time is {hour[0]} {mins} {prompt[0]}"

        if mode=='time':
            return dTime,hour[0], mins, prompt[0]
        elif mode=='date':
            return today
        else:
            return 'invalid prompt'


    # Convert audio file to text using the Whisper ASR API
    def audio2text(self, audio_file_path):
        with open(audio_file_path, 'rb') as f:
            audio_data = f.read()
        response = openai.Transcription.create(
            audio=audio_data,
            model="whisper",
            language="en-US"
        )
        return response['text']

    # Use the transcribed text as a prompt to generate response from ChatGPT API
    def getPromptResponce(self, prompt):
        openai.api_key=config.API_KEY
        # Generate a response from ChatGPT-3
        response = openai.Completion.create(
            engine='text-davinci-002',  # Specify the engine to use
            prompt=prompt,
            max_tokens=100,  # Set the maximum number of tokens for the response
            n=1,  # Specify the number of completions to generate
            stop=None,  # Specify a stop sequence to end the response (optional)
            temperature=0.7  # Set the temperature for randomness in the response
        )
        return response.choices[0].text.strip()
    
    # Convert text response to audio using gTTS
    def text2audio(self, text):
        tts = gTTS(text)
        audio_file_path = 'response.mp3'
        tts.save(audio_file_path)
        return audio_file_path

    # Play audio response to the user
    def play_audio(self, audio_file_path):
        playsound(audio_file_path)

