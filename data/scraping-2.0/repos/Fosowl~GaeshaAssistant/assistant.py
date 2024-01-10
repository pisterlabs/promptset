#!/usr/bin python3

import os
import sys
import time
from openai import error
from colorama import Fore
from datetime import datetime
import signal
import argparse
import subprocess

os.environ["SUNO_USE_SMALL_MODELS"] = "True"
os.environ["SUNO_ENABLE_MPS"] = "True" # turn to false if not using a macbook

# local

from sources.brain import setup_world, display_gpt_error, get_gpt_answer
from sources.interaction import get_user_query, emit_gpt_reply
from sources.system import expand_system_to_feed
from sources.speech import speech

# PARSER

parser = argparse.ArgumentParser(description='Ganesha, an AI assistant')
parser.add_argument('--silent', action='store_true',
                help='Prevent AI from speaking, only display answer')
parser.add_argument('--deaf', action='store_true',
                help='Do not listen to user, use text input instead')
parser.add_argument('--expertise', help='expertise field.', default="tech")
args = parser.parse_args()

AI_SPEAK = not args.silent
USER_SPEAK = not args.deaf

AI_NAME = "Gaesha"
GPT_MODEL = "gpt-4"
EXPERTISE = "python"
USER_FILE = "user.txt"
NATURAL_VOICE = False # require a very good GPU
STOP_KEYWORDS = ["goodbye", "ciao", "au revoir", "stop", "quit", "exit", "leave", "done", "finish", "terminate", "end", "close", "shut down", "shutup", "shut up", "shut-off", "shut off", "shut-down", "shut down", "turn off", "turn-off", "turnoff", "power off", "power-off", "poweroff", "kill", "abort"]
COMMANDS = {"SPEECH_ON": "(SPEECH_ON)",
            "SPEECH_OFF": "(SPEECH_OFF)",
            "STOP_LISTENING": "(STOP_LISTENING)",
            "START_LISTENING": "(START_LISTENING)"}

# GENERAL
max_api_token = 4097
max_reply_token = 1000

mic_process = None

interupt_request = 0
def handleInterrupt(signum, frame):
    global mic_process
    global interupt_request
    mic_process.terminate()
    if interupt_request >= 3:
        print(Fore.YELLOW + "Program aborted by user.")
        sys.exit(1)
    else:
        interupt_request += 1

def contain(sequence, keywords) -> bool:
    for key in keywords:
        if key.lower() in sequence.lower():
            return True
    return False

def get_user_description():
    user = ""
    try:
        with open(USER_FILE, 'r') as f:
            user = f.read()
    except FileNotFoundError as e:
        print(Fore.YELLOW + "User description file not provided.")
        pass
    return user

def get_today_goal(voice, gpt_model):
    goal = ""
    if AI_SPEAK:
        voice.say("Hello sir, what project are we working on today ?", 1) 
    else:
        print("Hello sir, What project are we working on today ?")
    goal = get_user_query(USER_SPEAK, gpt_model)
    if AI_SPEAK:
        voice.say("Sure, I will now assist you. I am setting up wait a bit...", 1)
    print("Sure, I will now assist you. setting up wait a bit")
    return goal

def user_instruction():
    print("Remember:")
    print("You can ask me to stop speaking by saying : " + COMMANDS["SPEECH_OFF"])
    print("You can ask me to speak by saying : " + COMMANDS["SPEECH_ON"])
    print("You can ask me to stop listening by saying : " + COMMANDS["STOP_LISTENING"])
    print("You can ask me to start listening by saying : " + COMMANDS["START_LISTENING"])
    print("You can stop at anytime by saying one of the following : ", STOP_KEYWORDS, " followed by okay")
    print("You can use tab to switch between gpt models")

def check_gpt_switch(feed, gpt_model):
    if "switch" in feed.lower():
        if gpt_model == "gpt-4" and "gpt-3" in feed.lower():
            gpt_model = "gpt-3.5-turbo"
        else:
            gpt_model = "gpt-4" and "gpt-4" in feed.lower()
    return gpt_model

def main():
    user_instruction()
    gpt_model = GPT_MODEL
    global mic_process
    voice = None
    if args.silent == True:
        print(Fore.LIGHTYELLOW_EX + "Silent mode enabled")
    else:
        print("loading natural voice model, this might take some time..." if NATURAL_VOICE else "ready to go!")
    voice = speech(load_natural=NATURAL_VOICE)
    mic_process = subprocess.Popen(["python3", "./sources/microphone/start.py"])
    time.sleep(1)
    if args.deaf == True:
        print(Fore.LIGHTYELLOW_EX + "Deaf mode enabled")
    conversation = setup_world(user_infos=get_user_description(),
                               ai_name=AI_NAME,
                               expertise=args.expertise,
                               goal=get_today_goal(voice, gpt_model=gpt_model),
                               commands=COMMANDS)
    signal.signal(signal.SIGINT, handler=handleInterrupt)
    while True:
        feed = get_user_query(user_speak=USER_SPEAK, gpt_model=gpt_model)
        feed = expand_system_to_feed(feed)
        gpt_model = check_gpt_switch(feed, gpt_model)
        if contain(feed, STOP_KEYWORDS) and len(feed.split(' ')) < 3:
            emit_gpt_reply(voice, "goodbye", commands=COMMANDS,
                                             user_speak=USER_SPEAK,
                                             ai_speak=AI_SPEAK)
            break
        try:
            answer = get_gpt_answer(feed, conversation, model=GPT_MODEL)
        except Exception as gpt_err:
            display_gpt_error(gpt_err)
            mic_process.terminate()
            if gpt_err == error.Timeout: 
                time.sleep(10)
                pass
            sys.exit(1)
        emit_gpt_reply(voice, answer,
                       commands=COMMANDS,
                       user_speak=USER_SPEAK, 
                       ai_speak=AI_SPEAK)
    mic_process.terminate()

if __name__ == "__main__":
    main()
