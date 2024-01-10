"""
FILE: test_with_openai.py

Uses openai and multithreading in python to interpret multiple solutions to a single question.

@author Noah Schliesman
Email: nschliesman@sandiego.edu
"""
import openai
from pocketsphinx import LiveSpeech
import threading as t
import time
from colorama import Fore
import gtts
import os
from playsound import playsound

#Global Variables
openai.api_key = ""
terminators = ["stop", "end"]
speak_options = ["speak", "Speak", "Talk", "talk", "voice", "Voice", "s", "S"]
type_options = ["type", "Type", "Write", "write", "t", "T"]
affirmations = ["yes", "Yes", "y", "Y"]
negations = ["no", "No", "n", "N"]

"""
Interprets user input based on speech or text input.

@return phrase the desired text input
"""
def interpret():
    print(Fore.GREEN+"Welcome to OpenAI, an engine for developing solutions to all of your problems\n\n")
    while True:
        mode = input("Would you like to speak into the engine or type: "+Fore.BLUE)
        if (mode in speak_options):
            start = time.time()
            for phrase in LiveSpeech():
                print(phrase)
                end = time.time()
                elapsed = end - start #Time elapsed during query
            if ((phrase in terminators) or (elapsed >= 2)):
                return (phrase, 1)
        elif (mode in type_options):
            multithreads = input(Fore.GREEN+"Would you like to run multiple processes: "+Fore.BLUE)
            if (multithreads in affirmations):
                thread_count = int(input(Fore.GREEN+"How many processes: "+Fore.BLUE))
            else:
                thread_count = 1
            phrase = input(Fore.GREEN+"What would you like to ask: "+Fore.BLUE)
            return (phrase, thread_count)
        else:
            print("Please enter a valid input.\n")

"""
Processes user input using OpenAI engine.

@param prompt string to process
@param temp float between 0 and 1 of engine certainty
@param max_token maximum tokens to output

@return text string that has been processed
"""
def process(prompt, temp, max_token):
    unformatted = openai.Completion.create(model="text-davinci-003", prompt=str(prompt), temperature=temp, max_tokens=max_token)
    text = unformatted.choices[0].text.lstrip() #strip text of junk
    return text

"""
Outputs audio using text to speech.

@param text the text to read

@return 0 arbitrary return value
"""
def audio_output(text):
    audio = gtts.gTTS(text=output, lang="en", slow=False) #handles audio
    filename = "f.mp3"
    audio.save(filename)
    playsound(filename)
    os.remove(filename)
    return 0

if __name__ == '__main__':
    interpreter, threads = interpret()
    if (threads == 1):
        output = process(interpreter,1,1500) #process threads
        print(Fore.MAGENTA+"\n"+output)
        audio_output(output)
    else:
        print("Feature not yet implemented")
