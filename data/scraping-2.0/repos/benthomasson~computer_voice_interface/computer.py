#!/usr/bin/env python -u

"""
Usage:
    computer [options]

Options:
    -h, --help          Show this page
    --debug             Show debug logging
    --verbose           Show verbose logging
    --prompt=<prompt>   Prompt to use
"""
from docopt import docopt
import logging
import sys
import os
import speech_recognition as sr
import pyaudio
import time
import openai
import subprocess
import string
from computer_fsm import ComputerFSM

# Replace YOUR_API_KEY with your OpenAI API key
openai.api_key = os.environ.get('API_KEY')


# Set the model and prompt
model_engine = "text-davinci-003"
prompt = "Hello"

# Set the maximum number of tokens to generate in the response
max_tokens = 128


logger = logging.getLogger('computer')


def play(audio_data):

    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=8000, output=True)
    stream.write(audio_data.frame_data)
    stream.stop_stream()
    stream.close()
    p.terminate()


def say(text):
    print(text)
    try:
        subprocess.run(["say", text])
    except KeyboardInterrupt:
        pass


def parse_args(args):
    parsed_args = docopt(__doc__, args)
    if parsed_args['--debug']:
        logging.basicConfig(level=logging.DEBUG)
    elif parsed_args['--verbose']:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.WARNING)
    return parsed_args


def recognize_audio(r, source):
    # read the audio data from the default microphone
    print("Ready")
    try:
        audio_data = r.listen(source, timeout=5)
        # play(audio_data)
        # convert speech to text
        text = r.recognize_whisper(audio_data)
    except sr.WaitTimeoutError:
        text = ""
    return text


def generate_response(prompt):

    # Generate a response
    try:
        completion = openai.Completion.create(
            engine=model_engine,
            prompt=prompt,
            max_tokens=1024,
            temperature=0.5,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        print(completion.choices)
        # Print the response
        say(completion.choices[0].text)
    except KeyboardInterrupt:
        pass


def main(args=None):
    if args is None:
        args = sys.argv[1:]
    parsed_args = parse_args(args)

    prompt_file = parsed_args['--prompt']
    prompt = None
    if prompt_file and os.path.exists(prompt_file):
        with open(prompt_file, 'r') as f:
            prompt = f.read()

    fsm = ComputerFSM({'prompt': prompt})

    r = sr.Recognizer()
    with sr.Microphone(sample_rate=8000) as source:
        r.adjust_for_ambient_noise(source, duration=5)
        while True:
            try:
                text = recognize_audio(r, source)
            except KeyboardInterrupt:
                return

            text = text.strip()
            text = text.lower()
            # remove all punctuation
            text = text.translate(str.maketrans('', '', string.punctuation))
            if text == "":
                continue

            if text == "15 15 15 15 15 15 15":
                continue

            print(f"You said '{text}'")
            # ok = input("Is this correct? [y/n] ")
            # if ok == "n":
            #    continue

            fsm.run(text)

            # generate_response(text)
            time.sleep(1)

    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
