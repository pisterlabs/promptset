import plotext
import numpy as np
import pyaudio
import struct
import wave
import time

from audio_get_channels import get_cur_mic
from audio_get_channels import get_speaker
from scipy.fftpack import fft
import openai
import credentials
import os
import pyttsx3
import threading
import sys

from rich import print
from rich.progress import track
from rich.console import Console
from rich.columns import Columns
from rich.panel import Panel

console = Console()

# ----------------------------------------------------------------
console.print(f'''\n...  RECORDING AUDIO \n\n''', style="bold red")
print(f'''...  Testline1\n''')
print(f'''...  Testline2\n''')
print(f'''...  Testline3\n''')

script_start = time.time()
script_dir = os.path.dirname(os.path.abspath(__file__))
filename = os.path.join(script_dir, "audio_output.wav")

# ----------------------------------------------------------------
def audio_spectrum(num_seconds):
    script_dir = os.path.dirname(os.path.abspath(__file__))

    chunk = 2205
    channels = 1
    fs = 44100
    seconds = num_seconds
    sample_format = pyaudio.paInt16
    filename = os.path.join(script_dir, "audio_output.wav")

    console.print(f'\n... Recording {seconds} seconds of audio initialized ...\n')

    p = pyaudio.PyAudio()
    stream = p.open(format=sample_format,
                    channels=channels,
                    rate=fs,
                    input_device_index=get_cur_mic(),
                    frames_per_buffer=chunk,
                    input=True)


    x = np.arange(0, chunk)
    x_fft = np.linspace(0, fs / 2, chunk // 2 + 1)

    frames = []
    start_time = time.time()


    while time.time() - start_time < seconds:
        plotext.clear_terminal(lines=10)
        plotext.clear_data()
        plotext.clear_figure()
        plotext.clear_color()

        # plotext.clt()  # to clear the terminal
        # plotext.cld()    # to clear the data onl
        # plotext.clf()    # to clear the figure
        # plotext.clc()    # to clear color

        data = stream.read(chunk, False)
        frames.append(data)
        data_int = struct.unpack(str(2 * chunk) + 'B', data)
        data_np = np.array(data_int, dtype='b')[::2] + 128

        y_freq = data_np
        spec = fft(data_int)
        y_spec = np.abs(np.fft.rfft(data_int)) / chunk

        # plotext.subplots(2, 1)
        # plotext.subplot(1, 1)
        plotext.plot(x, y_freq, color="white", marker="braille")
        plotext.title(f'[ {round(seconds - (time.time() - start_time), 1)}s | {seconds}s ]')
        # marker braille, fhd, hd, sd, dot, dollar,euro, bitcoin, at, heart, smile, queen, king,

        plotext.plot_size(200, 10)
        plotext.ylim(0, 300)

        plotext.yfrequency(2)
        plotext.xfrequency(0)
        plotext.xlim(0, 2205)
        plotext.horizontal_line(128, color="red", yside="top")

        # plotext.subplot(2, 1)
        # plotext.plot_size(200, 15)
        # plotext.plot(x_fft, y_spec, color="white", marker="braille")
        # plotext.ylim(0, 1)
        # plotext.xfrequency(2)
        # plotext.yfrequency(2)
        # plotext.xaxes("log")
        plotext.show()


    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    p.terminate()

    # Save the recorded data as a WAV file
    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(sample_format))
    wf.setframerate(fs)
    wf.writeframes(b''.join(frames))
    wf.close()


try:
    audio_spectrum(6)
except KeyboardInterrupt:
    pass

# -----------------------------------------------------------------------------------------
script_check_1 = time.time()
check_1 = script_check_1 - script_start
console.print(f'\n... Time: {round(check_1, 3)} seconds | Recording finished succesfully! \n')
# -----------------------------------------------------------------------------------------

def get_transcript_whisper():
    openai.api_key = credentials.api_key
    file = open(filename, "rb")
    transcription = openai.Audio.transcribe("whisper-1", file, response_format="json")
    transcribed_text = transcription["text"]
    return transcribed_text


text = get_transcript_whisper()

# If text contains one word of a stopwordlist then the script will stop
if any(word in text for word in ['stop', 'Stop', 'exit', 'quit', 'end', 'No', 'no']):
    print(f'''\n... SKRIPT STOPPED BY STOPWORD\n''')
    print(f'''__________________________________________________________________________________________________\n\n''')
    exit()



# -----------------------------------------------------------------------------------------
script_check_2 = time.time()
check_2 = script_check_2 - script_start
console.print(f'\n... Time: {round(check_2, 3)} seconds | Registered text: \n')
# ----------------------------------------------------------------
text = f' {text}'

def print_transcript():
    for word in text.split():
        time.sleep(0.27)
        print(word, end=' ', flush=True)

print_thread = threading.Thread(target=print_transcript())
print_thread.start()

def run_chatGPT(prompt):
    '''Run chatGPT with the prompt and return the response'''
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    answer = completion.choices[0].message.content

    return answer

answer = run_chatGPT(text)

# -----------------------------------------------------------------------------------------
script_check_3 = time.time()
check_3 = script_check_3 - script_start
console.print(f'\n\n... Time: {round(check_3, 3)} seconds | ChatGPT Answer: \n')
# -----------------------------------------------------------------------------------------
answer = f'{answer}'


# Initialize the pyttsx3 engine
engine = pyttsx3.init()
engine.setProperty('rate', 110)


# Define a function for speaking the anser
def speak_answer():
    engine.say(answer)
    engine.runAndWait()

# Define a function for printing the transcript
# Print function to slowly print the text except when the user presses Enter
# With pressing enter, the full answer is printed instantly
from rich import print
from rich.console import Console
from pynput import keyboard
import time
import sys

console = Console()

def print_answer(answer):
    print_complete = False
    break_program = False

    def on_press(key):
        nonlocal break_program
        if key == keyboard.Key.enter:
            console.print('Printout activated', style='bold red')
            break_program = True
            return False

    listener_thread = keyboard.Listener(on_press=on_press)
    listener_thread.start()

    try:
        for line in answer.splitlines():
            for word in line.split():
                console.print(word, end=' ')
                sys.stdout.flush()
                time.sleep(0.30)
                if break_program:
                    break

            sys.stdout.write('\n')

            if line == answer.splitlines()[-1] and word == line.split()[-1]:
                console.print('\nPrintout completed', style='bold green')
                print_complete = True
                break_program = True
            if print_complete:
                break

    finally:
        listener_thread.join()



# Create threads for speaking and printing the transcript
speak_thread = threading.Thread(target=speak_answer)
#print_thread = threading.Thread(target=print_transcript)

# Start both threads
#print_thread.start()
speak_thread.start()

# Wait for both threads to finish
#threading.wait_for(lambda: not speak_thread.is_alive()and not print_thread.is_alive())

# Wait for both threads to finish
#speak_thread.join()
#print_thread.join()
print_answer(answer)

# -----------------------------------------------------------------------------------------
script_check_4 = time.time()
check_4 = script_check_4 - script_start
console.print(f'''\n\n... Time: {round(check_4, 3)} seconds | Chat finished! \n''')

# -----------------------------------------------------------------------------------------
time.sleep(2)

# Restart the script
# while True:

    # Restart the program
    # python = sys.executable
    #os.execl(python, python, *sys.argv, )


