#! /usr/bin/env python

import argparse
from datetime import datetime
import os
import pdb
import re
from openai_model import *
import speech_recognition as sr
from util import run_cmd, datestamp
import pyaudio

# Text-to-speech engine
# engine = pyttsx3.init()

CHUNK = 1024 * 2
FORMAT = pyaudio.paInt16
NPFORMAT = np.int16
CHANNELS = 1
RATE = 44100
#WAVE_OUTPUT_FILENAME = 'frompy.wav'

# Max readings**2 for a chunck of length 2048
#heuristic_max_power = 11846935
# Float
heuristic_max_power = 826346439192.0


def get_text_from_audio(args, saveto):
    # https://stackoverflow.com/questions/40704026/voice-recording-using-pyaudio
    pp = pyaudio.PyAudio()
    stream = pp.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK)

    print("*** recording ***")
    frames = []

    max_power = 0.0
    for ii in range(int(RATE / CHUNK * 10)): #args.seconds)):
        data = stream.read(CHUNK)
        frames.append(data)
        npdat = np.frombuffer(data, dtype=np.int16).astype(np.float64)
        power = (npdat**2).sum()
        frac = min(1.0, power / heuristic_max_power)
        st = '*' * (1 + int(frac ** .125 * 60))
        #print(f'{ii}: {npdat.sum()}   {sum(data)} {power}')
        #print(f'{ii:02d}: {st}')
        if ii % 1 == 0:
            #print((npdat**2)[:12])
            print(f'{frac:.04f} {st}')
        max_power = max(power, max_power)

    print('Max power was: ', max_power)
    stream.stop_stream()
    stream.close()
    pp.terminate()

    wf = wave.open(saveto, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(pp.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    print(f'\nWrote file: {saveto}')

    audio_filename = saveto

    model = whisper.load_model(args.whisper_model)
    result = model.transcribe(audio_filename)
    text = result['text']
    return text

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model', type=str, default='gpt-3.5-turbo', help='openai model name') # text-ada-001
    parser.add_argument(
        '--prompt_name',
        type=str,
        default='test_prompt',
        help='path to the prompt file')
    parser.add_argument(
        '--test_input_name',
        type=str,
        default='test',
        help='path to the user input file')
    parser.add_argument(
        '--split_start',
        type=str,
        default='Answer: ',
        help='token indicating start of answer')
    parser.add_argument(
        '--split_end',
        type=str,
        default=None,
        help='token indicating end of answer')
    parser.add_argument(
        '--temperature', type=float, default=0.5, help='temperature')
    parser.add_argument(
        '--max_decoding_steps', type=int, default=128, help='max tokens')
    parser.add_argument(
        '--stop_token',
        type=str,
        default='',
        help='stop decoding on token')
    parser.add_argument(
        '--api_key', type=str, default=None, help='openai api key to use')
    parser.add_argument(
        '--interactive', action='store_true', help='Use chat or not')
    return parser


def main():
    parser = make_parser()
    args = parser.parse_args()
    if args.prompt_name == '':
        prompt_string = ''
    else:
        prompt_string = open(
            os.path.join('prompts', args.prompt_name + '.txt')).read()
        if '\\n' in prompt_string:
            prompt_string = re.sub('\\\\n', '\n', prompt_string)

    if args.test_input_name == '':
        test_input_string = 'I am worthless.'
    else:
        print("opening...")
        test_input_string = open(
            os.path.join('user_inputs', args.test_input_name + '.txt')).read()
        if '\\n' in test_input_string:
            test_input_string = re.sub('\\\\n', '\n', test_input_string)

    print("prompt string is {}".format(prompt_string))

    messages = [ {"role": "system", "content": 
              prompt_string} ]

    # Try to recognize the audio

    if args.interactive:
        while True:
            dt = datestamp()
            saveto = f'audio_{dt}.flac'
            message = get_text_from_audio(args, saveto)

            if message:
                messages.append(
                    {"role": "user", "content": message},
                )
                # print("appended  message {}".format(message))
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo", messages=messages
                )
                # print("chat output {}".format(chat))
    
                reply = response.choices[0].message.content
                print(f"ChatGPT: {reply}")
                messages.append({"role": "assistant", "content": reply})

                # Get the response text
                response_text = str(response['choices'][0]['text']).strip('\n\n')
                print(response_text)

                run_cmd(('say', '-v', 'Daniel', '-r', str(args.voice_rate), response_text))
    else:
        chat = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=messages
        )
        
        reply = chat.choices[0].message.content
        print(f"ChatGPT: {reply}")


if __name__ == '__main__':
    main()
