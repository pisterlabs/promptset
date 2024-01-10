from issai.asr import ASR
from issai.tts import TTS
from issai import utils
import openai
import subprocess
import telebot
import json
import os


# create directories to save
# input and output files
utils.make_dir("input/voice")
utils.make_dir("output")

# initialize telegram bot
isRunning = False
tele_token = "YOUR TELEGRAM BOT TOKEN"
tele_bot = telebot.TeleBot(tele_token, threaded=True)

# the first message to ChatGPT
messages = [
    # system message first, it helps set the behavior of the assistant
    {"role": "system", "content": "You are a helpful assistant."},
]
reset_chat = True # set False if you want ChatGPT to use chat history

# initialize ASR and TTS models
asr = ASR(lang='en', model='google') # to use offline vosk asr: 'google' -> 'vosk'
tts = TTS('google') # to use offline pyttsx3 tts: 'google' -> 'other'

@tele_bot.message_handler(commands=["start", "go"])
def start_handler(message):
    global isRunning
    if not isRunning:
        tele_bot.send_message(message.chat.id, "Welcome to Voice ChatGPT!")
        isRunning = True

@tele_bot.message_handler(content_types=['voice'])
def voice_processing(message):
    # process the voice message
    file_info = tele_bot.get_file(message.voice.file_id)
    file_data = tele_bot.download_file(file_info.file_path)

    raw_audio_path = './input/' + file_info.file_path
    with open(raw_audio_path, 'wb') as f:
        f.write(file_data)

    #tele_bot.reply_to(message, "Processing...")
    input_audio_path = raw_audio_path + ".wav"
    process = subprocess.run(['ffmpeg', '-i', raw_audio_path, input_audio_path])
    if process.returncode != 0:
        raise Exception("Something went wrong")

    # convert the audio input to text
    asr.convert(input_audio_path)

    tele_bot.reply_to(message, "You: " + asr.message)
    print("User:", asr.message)

    # send the message to ChatGPT
    if reset_chat:
        chat_completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=[{"role": "user", "content": asr.message}])
    else:
        messages.append({"role": "user", "content": asr.message},)
        chat_completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=messages)
        messages.append({"role": "assistant", "content": chat_completion.choices[0].message.content})

    response = chat_completion.choices[0].message.content
    print("ChatGPT:", response)

    # send the text response to telegram
    tele_bot.send_message(message.chat.id, response)

    # create a path to save the audio
    output_audio_path = os.path.join('output', 'answer.mp3')

    # convert the answer to speech
    tts.convert(response, output_audio_path)

    # send the voice response to the telegram
    tele_bot.send_voice(message.chat.id, voice=open(output_audio_path, "rb"))

    # remove input and output files
    os.remove(raw_audio_path)
    os.remove(input_audio_path)
    os.remove(output_audio_path)

# run the telegram bot
tele_bot.polling(none_stop=True)
