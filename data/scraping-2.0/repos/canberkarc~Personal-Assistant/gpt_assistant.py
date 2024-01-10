import openai
import asyncio
import re
import whisper
import boto3
import pydub
from pydub import playback
import speech_recognition as sr
from EdgeGPT import Chatbot, ConversationStyle

# Initialize the OpenAI API
openai.api_key = ""

# Create a recognizer object and wake word variables
recognizer = sr.Recognizer()
BING_WAKE_WORD = "bing"
GPT_WAKE_WORD = "gpt"

def get_wake_word(phrase):
    if BING_WAKE_WORD in phrase.lower():
        return BING_WAKE_WORD
    elif GPT_WAKE_WORD in phrase.lower():
        return GPT_WAKE_WORD
    else:
        return None
    
def synthesize_speech(text, output_filename):
    polly = boto3.client('polly', region_name='us-west-2',  aws_access_key_id='', aws_secret_access_key='')

    response = polly.synthesize_speech(
        Text=text,
        OutputFormat='mp3',
        VoiceId='Salli',
        Engine='neural'
    )

    with open(output_filename, 'wb') as f:
        f.write(response['AudioStream'].read())

def play_audio(file):
    sound = pydub.AudioSegment.from_file(file, format="mp3")
    playback.play(sound)

async def main():
    while True:

        model = whisper.load_model("base")
        print("Speak a prompt...")
        synthesize_speech('What can I help you with?', 'C:/Users/a/Desktop/personal_assistant/response.mp3')

        with sr.Microphone() as source:
            
            recognizer.adjust_for_ambient_noise(source)

            play_audio('C:/Users/a/Desktop/personal_assistant/response.mp3')
            audio = recognizer.listen(source)

            try:
                with open("C:/Users/a/Desktop/personal_assistant/audio_prompt.wav", "wb") as f:
                    f.write(audio.get_wav_data())

                result = model.transcribe("C:/Users/a/Desktop/personal_assistant/audio_prompt.wav", fp16=False)
                user_input = result["text"]
                print(f"You said: {user_input}")
            except Exception as e:
                print("Error transcribing audio: {0}".format(e))
                continue


            # Send prompt to GPT-3.5-turbo API
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content":
                    "You are a helpful assistant."},
                    {"role": "user", "content": user_input},
                ],
                temperature=0.5,
                max_tokens=150,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                n=1,
                stop=["\nUser:"],
            )

            bot_response = response["choices"][0]["message"]["content"]
                
        print("Bot's response:", bot_response)
        synthesize_speech(bot_response, 'C:/Users/a/Desktop/personal_assistant/response.mp3')
        play_audio('C:/Users/a/Desktop/personal_assistant/response.mp3')

if __name__ == "__main__":
    asyncio.run(main())