
from colorama import Fore
import os
import time
import openai
from sources.speech import *

CONFIRM_WORDS = ["understood", "thanks", "do it"]
RESET_WORDS = ["reset", "I repeat", "nevermind", "I said", "I mean", "I am saying", "I am saying", "I'm saying", "hey" "I am back", "wake up", "can you"]
# openai whisper hallucinations
HALLUCINATIONS = ["for watching", "please subscribe", "enjoyed this video", "subscribe to my channel", "thumbs up", "Bye-bye", "I'll see you in the next video"]

# RECORD
wav_path = "./record.wav"

# COLOR
REPLY_COLOR_TEXT = Fore.LIGHTCYAN_EX
REPLY_COLOR_CODE = Fore.LIGHTBLUE_EX
ENTRY_COLOR = Fore.LIGHTGREEN_EX

def execute_gpt_command(word, commands, user_speak, ai_speak):
    command_keyword = "[COMMAND]"
    if command_keyword not in word:
        return False
    if commands["SPEECH_OFF"] in word:
        print(Fore.YELLOW + "(SPEECH OFF)")
        ai_speak = False
        return True
    if commands["SPEECH_ON"] in word:
        print(Fore.YELLOW + "(SPEECH ON)")
        ai_speak = True
        return True
    if commands["STOP_LISTENING"] in word:
        print(Fore.YELLOW + "(SWITCH TO TEXT INPUT)")
        user_speak = False
        return True
    if commands["START_LISTENING"] in word:
        print(Fore.YELLOW + "(SWITCH TO VOICE INPUT)")
        user_speak = True
        return True
    return False

def emit_gpt_reply(voice, answer, commands, user_speak, ai_speak) -> None:
    if answer == None:
        return
    # text
    to_display_string = REPLY_COLOR_TEXT
    sayable = ""
    on_code_block = False
    color_trigger = ""
    for word in answer.split(' '):
        if color_trigger != "":
            to_display_string += color_trigger
            color_trigger = ""
        if execute_gpt_command(word, commands, user_speak, ai_speak) == True:
            continue
        if "```" in word and on_code_block == False:
            color_trigger = REPLY_COLOR_CODE + word
            on_code_block = True
        elif "```" in word and on_code_block == True:
            color_trigger = word + REPLY_COLOR_TEXT
            on_code_block = False
        else:
            to_display_string += word + " "
        if not on_code_block:
            sayable += word + " "
    to_display_string = to_display_string.replace("`", "")
    print(to_display_string)
    if (ai_speak) == True and voice:
        voice.say(sayable, 6)

def parse_out_hallucinations(interpretation):
    for h in HALLUCINATIONS:
        if h.lower() in interpretation.lower():
            print(Fore.YELLOW, "Whisper hallucination parsed out.")
            return ""
    return interpretation

def contain(sequence, keywords) -> bool:
    for key in keywords:
        if key.lower() in sequence.lower():
            return True
    return False

def is_file_bad(file_path):
    try:
        with open(file_path, 'rb') as f:
            return len(f.read()) < 1024
    except FileNotFoundError:
        return True

def transcript_audio(user_speak, gpt_model):
    transcript = ""
    done = False
    last_transcript = ""
    while not done:
        if is_file_bad('./record.wav'):
            print(Fore.YELLOW, "waiting for subprocess", Fore.WHITE)
            time.sleep(2)
            continue
        start_listen_time = time.time()
        audio_file = open("./record.wav", "rb")
        whisper_interpretation = openai.Audio.translate("whisper-1", audio_file, temperature=0)
        if whisper_interpretation == last_transcript:
            continue
        last_transcript = whisper_interpretation
        end_listen_time = time.time()
        comprehension_time = round(end_listen_time - start_listen_time, 1)
        whisper_interpretation = parse_out_hallucinations(whisper_interpretation['text'])
        print(Fore.LIGHTBLACK_EX, f"Understand : {whisper_interpretation} ({comprehension_time} s)")
        transcript += whisper_interpretation
        if contain(whisper_interpretation.lower(), CONFIRM_WORDS):
            done = True
        if contain(whisper_interpretation.lower(), RESET_WORDS):
            print(Fore.YELLOW, "Waiting for reformulated query")
            transcript = ""
    return transcript

def cleanup_mess():
    try:
        os.remove('record.wav')
        os.remove('tmp.wav')
    except:
        pass

def get_user_query(user_speak, gpt_model) -> str:
    cleanup_mess()
    if user_speak == True:
        print(ENTRY_COLOR + f">>> Listening... <<<({gpt_model})")
        transcript = transcript_audio(user_speak, gpt_model)
        return transcript
    else:
        buffer = ""
        while buffer == "" or buffer.isascii() == False:
            buffer = input(ENTRY_COLOR + f"{gpt_model}>>> ")
        return buffer