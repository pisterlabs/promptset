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
openai.api_key = "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"

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
    polly = boto3.client('polly', region_name='ap-south-1')
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

        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source)
            print(f"Waiting for wake words 'bing' or 'GPT'...")
            while True:
                audio = recognizer.listen(source)
                try:
                    with open("audio.wav", "wb") as f:
                        f.write(audio.get_wav_data())
                    # Use the preloaded tiny_model
                    model = whisper.load_model("tiny")
                    result = model.transcribe("audio.wav")
                    phrase = result["text"]
                    #phrase = BING_WAKE_WORD
                    print(f"You said: {phrase}")

                    wake_word = get_wake_word(phrase)
                    if wake_word is not None:
                        break
                    else:
                        print("Not a wake word. Try again.")
                except Exception as e:
                    print("Error transcribing audio: {0}".format(e))
                    continue

            print("Speak a prompt...")
            synthesize_speech('What can I help you with?', 'response.mp3')
            play_audio('response.mp3')
            audio = recognizer.listen(source)

            try:
                with open("audio_prompt.wav", "wb") as f:
                    f.write(audio.get_wav_data())
                model = whisper.load_model("base")
                result = model.transcribe("audio_prompt.wav")
                user_input = result["text"]
                print(f"You said: {user_input}")
            except Exception as e:
                print("Error transcribing audio: {0}".format(e))
                continue

            if wake_word == BING_WAKE_WORD:
                bot = Chatbot(cookiePath='cookies.json')
                response = await bot.ask(prompt=user_input, conversation_style=ConversationStyle.precise)

                for message in response["item"]["messages"]:
                    if message["author"] == "bot":
                        bot_response = message["text"]

                bot_response = re.sub('\[\^\d+\^\]', '', bot_response)
                # Select only the bot response from the response dictionary
                for message in response["item"]["messages"]:
                    if message["author"] == "bot":
                        bot_response = message["text"]
                # Remove [^#^] citations in response
                bot_response = re.sub('\[\^\d+\^\]', '', bot_response)

            else:
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
                    stop=["/nUser:"],
                )

                bot_response = response["choices"][0]["message"]["content"]

        print("Bot's response:", bot_response)
        synthesize_speech(bot_response, 'response.mp3')
        play_audio('response.mp3')
        await bot.close()

if __name__ == "__main__":
    asyncio.run(main())

    '''

import openai_secret_manager
import openai
import speech_recognition as sr
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play as playback
import pyaudio
import asyncio

# Set up OpenAI API credentials
assert "openai" in openai_secret_manager.secrets.get_services()
secrets = openai_secret_manager.get_secret("openai")
openai.api_key = secrets["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"]

# Set up speech recognition
r = sr.Recognizer()

# Set up PyAudio
pa = pyaudio.PyAudio()

async def generate_response(prompt):
    # Use OpenAI API to generate response
    response = await openai.Completion.create(
        engine="davinci", prompt=prompt, max_tokens=60
    )

    # Extract the bot response from the API response
    bot_response = ""
    for message in response["choices"][0]["text"].split("\n"):
        if message.startswith("bot:"):
            bot_response = message.replace("bot:", "").strip()

    return bot_response


def play_audio(audio_file_path):
    # Load audio file using PyDub
    sound = AudioSegment.from_file(audio_file_path)

    # Play audio using PyDub
    playback(pa, sound)

async def main():
    # Use PyAudio to record audio from the user
    print("Speak now")
    with sr.Microphone() as source:
        audio = r.listen(source)
        with open("input.wav", "wb") as f:
            f.write(audio.get_wav_data())

    # Transcribe user's speech to text using Google Speech Recognition
    print("Transcribing...")
    with sr.AudioFile("input.wav") as source:
        audio_text = r.record(source)
    try:
        user_input = r.recognize_google(audio_text)
        print("You:", user_input)
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
        return
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
        return

    # Generate response using OpenAI API
    print("Generating response...")
    prompt = f"User: {user_input}\nBot:"
    bot_response = await generate_response(prompt)
    print("Bot's response:", bot_response)

    # Convert bot response to audio using gTTS
    tts = gTTS(text=bot_response, lang="en")
    tts.save("response.mp3")

    # Play audio response to user
    print("Playing audio response...")
    play_audio('response.mp3')


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(main())
    finally:
        loop.close()
        pa.terminate()
'''
