"""1) Listen for speech prompts to send to an LLM; 2) Interpret LLM response to control bot"""

import curses
import os
import queue
import string
import sys
import time
import threading
import wave

import numpy as np
import openai
import pyttsx3
import pyaudio
import serial
from serial.tools import list_ports

import ipdb


# <s Set up.
print(f"Setting up...")

# Define some constants
wakeword = "hey wally"
ai_response = "yes, I'm listening..."
ai_silence_check = "hi, are you still there?"
pyaud_sr = 16000             # mic sampling rate
pyaud_chunk_sz = 4096        # n samples
n_pyaud_mic_channels = 2     # n mic channels
wakeword_det_dur = 3         # seconds
silence_check_dur = 2.5      # seconds
wakeword_wav = "wakeword_detect.wav"
convo_wav = "convo.wav"
gpt_model = "gpt-4"  # "gpt-3.5-turbo" or "gpt-4"
model_temp = 0.25
openai.api_key = "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"
global_rec = True

# Helper functions

# Create a string translator that removes punctuation
no_punc_trans = str.maketrans('', '', string.punctuation + '\n' + '\t' + '\r')

class TerminateProgram(Exception):
    """Terminates the program"""
    pass 


def check_silence(q, freq_ratio_thresh=0.3, lo_freq=80, hi_freq=400, silence_check_dur=silence_check_dur):
    """Used w/ threading: Checks for silence in an audio queue by looking at ratio of human vocal freqs"""
    silence_start_time = None
    global global_rec
    while global_rec:  # flag for currently recording audio (global var)
        if not q.empty():
            audio_data = q.get()
            audio_data = np.frombuffer(audio_data, dtype=np.int16)
            if len(audio_data.shape) == 2:
                audio_data = np.mean(audio_data, axis=1)
            amplitudes = np.abs(np.fft.rfft(audio_data))[1:]
            fft_freqs = np.fft.rfftfreq(audio_data.size, d=(1 / pyaud_sr))[1:]
            human_vocal_amps = amplitudes[np.logical_and((fft_freqs > lo_freq), (fft_freqs < hi_freq))]
            freq_ratio = np.sum(human_vocal_amps) / np.sum(amplitudes)
            if freq_ratio < freq_ratio_thresh:
                if silence_start_time is None:
                    silence_start_time = time.time()
                elif (time.time() - silence_start_time) > silence_check_dur:
                    #ipdb.set_trace()
                    break
            else:
                silence_start_time = None
    global_rec = False

# Serial
ports = list_ports.comports()
ser_port = sorted(ports)[-1].device
ser = serial.Serial(ser_port, 19200, timeout=1)

# TTS
pyttsx3_engine = pyttsx3.init()
pyttsx3_engine.setProperty("volume", 0.9)
pyttsx3_engine.setProperty("rate", 160)
pyttsx3_engine.setProperty("voice", "english-us")

# PyAudio
pyaud = pyaudio.PyAudio()
pyaud_dev_idx = None
for i in range(pyaud.get_device_count()):
    info = pyaud.get_device_info_by_index(i)
    if "NB3_ear_card" in info["name"]:
        pyaud_dev_idx = i
        break
mic_stream = pyaud.open(
    format=pyaudio.paInt16,
    channels=n_pyaud_mic_channels,
    rate=pyaud_sr,
    frames_per_buffer=pyaud_chunk_sz,
    input_device_index=pyaud_dev_idx,
    input=True,
)

# # Curses
# stdscr = curses.initscr()    # init curses screen
# stdscr.keypad(True)          # enable curses keypad
# curses.cbreak()              # respond immediately to keypress
# curses.noecho()              # echo keypress to screen
# curses.curs_set(True)        # make cursor visible
# /s>

# <s Listen for keyword detection
n_frames_wakeword_det = int(pyaud_sr / pyaud_chunk_sz * wakeword_det_dur)
while True:
    # GPT system prompt
    conversation = [
        {"role": "system", 
        "content": (
            "You are a small two-wheeled robot. Your name is Wally. Your task is to "
            "make a two-part response to each of my requests to you, which will ask you "
            "to move in a particular way. The first part of your response should be a "
            "reply that explains in natural language how you would move. This reply should "
            "be sensible, but also somewhat funny and snarky. You should always start your "
            "reply with 'HIGH' if you are excited about my request, and 'LOW' if you are "
            "not excited about my request. You should always finish this first part "
            "of your response with '##'. The second part of your response should "
            "contain the commands necessary for you to move in a particular way. You have "
            "four commands for moving: you can move forwards, you can move backwards, you "
            "can turn left, and you can turn right. To move forwards, you respond with "
            "'w', to move backwards you respond with 's', to turn left you respond with "
            "'a', and to turn right you respond with 'd'. Additionally, you must specify "
            "for how many milliseconds you'll move in each direction. You must start this second part of your response immediately with the commands - there should be no preface to the commands. For example, if I "
            "ask you to create a square with your movement, you should respond with "
            "something like: 'w3000 d500 w3000 d500 w3000 d500 w3000'. This consists of "
            "seven movements. You should NEVER have more than 20 movements in a response. "
            "This particular set of seven movements would make you "
            "move in a straight line for 3 seconds, then turn 90 degrees to the right, "
            "then move in a straight line for 3 seconds, then turn 90 degrees to the "
            "right, then move in a straight line for 3 seconds, then turn 90 degrees to "
            "the right, then move in a straight line for 3 seconds, thus creating a square "
            "with your movement. Similarly, if I ask you to create an equilateral triangle "
            "with your movement, you should respond with something like: "
            "'d250 w3000 d500 w3000 d250 w3000.' As a last example, if I tell you to move "
            "like a snake, then maybe you would respond with something like: "
            "'w2000 d250 d1000 d250 w1000 a250 w500 a250 w2000 d250 w1000 d250 d3500 d250 "
            "w2000'. This second part of your respone should ALWAYS and ONLY be in this "
            "format of '<first_movement_letter><first_movement_duration_ms> "
            "<second_movement_letter><second_movement_duration_ms> ... "
            "<last_movement_letter><last_movement_duration_ms>', and "
            "remember, you should NEVER have more than 20 movements in a response."
            )
        }
    ]
    response = openai.ChatCompletion.create(
        model=gpt_model,
        temperature=model_temp,
        messages=conversation
    )
    conversation.append(
        {
            "role": "assistant",
            "content": response["choices"][0]["message"]["content"]
        }
    )
    print(f"Set-up complete.  Ready to talk!")
    wakeword_detected = False
    try:
        while not wakeword_detected:
            time.sleep(0.1)
            # Check to see if quit.
            # char = chr(stdscr.getch())
            # if char == 'q':  # clean-up and quit
            #     # Close serial.
            #     ser.close()
            #     # Close pyaudio.
            #     mic_stream.stop_stream()
            #     mic_stream.close()
            #     pyaud.terminate()
            #     # Stop pyttsx3.
            #     pyttsx3_engine.stop()
            #     # Quit curses.
            #     curses.nocbreak()
            #     stdscr.keypad(False)
            #     curses.endwin()
            #     raise TerminateProgram
            # Get audio data, chunk by chunk
            frames = []
            mic_stream.start_stream()
            for i in range(n_frames_wakeword_det):
                data = mic_stream.read(pyaud_chunk_sz, exception_on_overflow=False)
                frames.append(data)
            mic_stream.stop_stream()
            # Save as wav file
            wf_wakeword = wave.open(wakeword_wav, mode="wb")
            wf_wakeword.setnchannels(n_pyaud_mic_channels)
            wf_wakeword.setsampwidth(2)  # 2 bytes
            wf_wakeword.setframerate(pyaud_sr)
            wf_wakeword.setnframes(n_frames_wakeword_det)
            wf_wakeword.writeframes(b''.join(frames))
            wf_wakeword.close()
            # Send to whisper
            transcription = openai.Audio.transcribe("whisper-1", open(wakeword_wav, "rb"))
            print(transcription["text"])
            txt = transcription["text"].lower().translate(no_punc_trans)
            if wakeword in txt:
                print(ai_response)
                pyttsx3_engine.say(ai_response)
                pyttsx3_engine.runAndWait()
                wakeword_detected = True
    except TerminateProgram:
        pass
    # /s>
    # <s Start conversation
    conversing = True
    while conversing:
        # Set up thread for silence checks
        audio_q = queue.Queue()
        silence_thread = threading.Thread(target=check_silence, args=(audio_q,))
        silence_thread.start()
        mic_stream.start_stream()
        frames = []
        while global_rec:
            data = mic_stream.read(pyaud_chunk_sz, exception_on_overflow=False)
            frames.append(data)
            audio_q.put(data)
            # if len(frames) > 100:
            #     global_rec = False
            #     print(len(frames))
        global_rec = True
        silence_thread.join()
        mic_stream.stop_stream()
        # Save as wav file
        wf_convo = wave.open(convo_wav, mode="wb")
        wf_convo.setnchannels(n_pyaud_mic_channels)
        wf_convo.setsampwidth(2)  # 2 bytes
        wf_convo.setframerate(pyaud_sr)
        wf_convo.writeframes(b''.join(frames))
        wf_convo.close()
        # Send to whisper
        transcription = openai.Audio.transcribe("whisper-1", open(convo_wav, "rb"))
        txt = transcription["text"].lower().translate(no_punc_trans)
        print(txt)
        #ipdb.set_trace()
        if "bye wally" in txt:
            #ipdb.set_trace()
            conversing = False
            conversation.append(
                {
                    "role": "user",
                    "content": txt
                }
            )
            response = openai.ChatCompletion.create(
                        model=gpt_model,
                        temperature=model_temp,
                        messages=conversation
                    )
            full_reply = response["choices"][0]["message"]["content"]
            print(full_reply)
            conversation.append(
                {
                    "role": "assistant",
                    "content": full_reply
                }
            )
            reply, cmds = response["choices"][0]["message"]["content"].split("##")
            cmds = cmds.translate(no_punc_trans)
            pyttsx3_engine.say(reply)
            pyttsx3_engine.runAndWait()
        # Send to GPT
        else:
            if len(txt) > 15:
                conversation.append(
                    {
                        "role": "user",
                        "content": txt
                    }
                )
                response = openai.ChatCompletion.create(
                    model=gpt_model,
                    temperature=model_temp,
                    messages=conversation
                )
                full_reply = response["choices"][0]["message"]["content"]
                print(full_reply)
                conversation.append(
                    {
                        "role": "assistant",
                        "content": full_reply
                    }
                )
                reply, cmds = response["choices"][0]["message"]["content"].split("##")
                #ipdb.set_trace()
                cmds = cmds.translate(no_punc_trans)
                pyttsx3_engine.say(reply)
                pyttsx3_engine.runAndWait()
                if "HIGH" in reply:
                    ser.write(b"h")
                if "LOW" in reply:
                    ser.write(b"l")
                cmds = cmds.split(" ")
                if cmds[0] == "":
                    cmds = cmds[1:]
                pyttsx3_engine.say(f"Get ready for {len(cmds)} moves!")
                pyttsx3_engine.runAndWait()
                for c in cmds:
                    ser.write(c[0].encode("utf-8"))
                    sleep_dur = int(c[1:]) / 1000
                    time.sleep(sleep_dur)
                ser.write(b"s")
                print("Ok, ready for your next command after 1 sec.")
                pyttsx3_engine.say("Ok, ready for your next command after 1 sec.")
                pyttsx3_engine.runAndWait()
                time.sleep(1)
                #/s>

# # -----

# # Start streaming audio
# stream.start_stream()

# # Append frames of data until key (spacebar) is pressed
# frames = []
# for i in range(0, int(RATE / CHUNK * MAX_DURATION)):
#     # Read raw data and append
#     raw_data = stream.read(CHUNK)
#     frames.append(raw_data)

#     # Check for key press ('z')
#     char = screen.getch()
#     if char == ord('z'):
#         break

# # Stop stream
# stream.stop_stream()

# # Write to WAV file
# wf.writeframes(b''.join(frames))

# # Close WAV file
# wf.close()
# # Send to whisper


# Initialize conversation history
# conversation = [
#     {"role": "system", "content": "You are small two wheeled robot. Your name is NB3, which stands for no black box bot. \
#      Your task is to respond to requests for you to move in a particular way with a sensible, funny, somwhat snarky text reply and a sequence of movements. \
#      The movement commands should follow immediately after a '##' at the end of your text reply. There should be a final '##' at the end of the commands. \
#      They should have the following format: \"<some text reply you produce>##f200 l300 r100 b75##\". \
#      The commands must consist of single letters (f,b,l,r) followed by a number. f is forward, b is backward, l is left turn, r is right turn, and the numbers \
#      indicate how long the robot should perform the movement for in milliseconds. So, for the previous example, the robot would move forward for 200 ms, make a \
#      left turn for 300 ms, a right turn for 100 ms, and go backward for 100 ms."},
# ]

# # Configure serial port
# ser = serial.Serial()
# ser.baudrate = 19200
# ser.port = '/dev/ttyUSB0'

# # Open serial port
# ser.open()
# time.sleep(1.50) # Wait for connection before sending any data

# # Robot initial state (waiting and stopped)
# ser.write(b'x')
# time.sleep(0.05)
# ser.write(b'w')
# time.sleep(0.05)

# # Initialize speech engine
# engine = pyttsx3.init()

# # Set sound recording format
# CHUNK = 1600                # Buffer size
# FORMAT = pyaudio.paInt16    # Data type
# CHANNELS = 1                # Number of channels
# RATE = 16000                # Sample rate (Hz)
# MAX_DURATION = 5            # Max recording duration
# WAVE_OUTPUT_FILENAME = "speech.wav"

# # Get pyaudio object
# pya = pyaudio.PyAudio()

# # Open audio stream (from default device)
# stream = pya.open(format=FORMAT,
#             channels=CHANNELS,
#             rate=RATE,
#             input=True,
#             start=False,
#             frames_per_buffer=CHUNK)

# # Setup the curses screen window
# screen = curses.initscr()
# curses.noecho()
# curses.cbreak()
# screen.nodelay(True)
 
# # --------------------------------------------------------------------------------
# # HELPER FUNCTIONS
# # --------------------------------------------------------------------------------

# # Function to record speech snippets to a WAV file
# def record_speech(stream):

#     # Prepare a WAV file
#     wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
#     wf.setnchannels(CHANNELS)
#     wf.setsampwidth(2)
#     wf.setframerate(RATE)

#     # Start streaming audio
#     stream.start_stream()

#     # Append frames of data until key (spacebar) is pressed
#     frames = []
#     for i in range(0, int(RATE / CHUNK * MAX_DURATION)):
#         # Read raw data and append
#         raw_data = stream.read(CHUNK)
#         frames.append(raw_data)
    
#         # Check for key press ('z')
#         char = screen.getch()
#         if char == ord('z'):
#             break

#     # Stop stream
#     stream.stop_stream()

#     # Write to WAV file
#     wf.writeframes(b''.join(frames))
    
#     # Close WAV file
#     wf.close()

#     return
# # --------------------------------------------------------------------------------


# # --------------------------------------------------------------------------------
# # Chat Loop
# # --------------------------------------------------------------------------------
# try:
#     while True:

#         # Wait to start talking
#         screen.addstr(0, 0, "Press 'z' to talk to your NB3 ('q' to quit):")
#         screen.clrtoeol()
#         while True:
#             char = screen.getch()
#             if char == ord('q'):
#                 sys.exit()
#             elif char == ord('z'):
#                 break

#         # Indicate hearing (stop moving and blink)
#         ser.write(b'x')
#         time.sleep(0.05)
#         ser.write(b'h')
#         time.sleep(0.05)

#         # Start recording
#         screen.addstr("...press 'z' again to stop speaking.", curses.A_UNDERLINE)
#         record_speech(stream)
#         screen.erase()        

#         # Indicate done hearing (twitch and wait)
#         ser.write(b'l')
#         time.sleep(0.15)
#         ser.write(b'x')
#         time.sleep(0.05)
#         ser.write(b'r')
#         time.sleep(0.15)
#         ser.write(b'x')
#         time.sleep(0.05)
#         ser.write(b'w')
#         time.sleep(0.05)

#         # Get transcription from Whisper
#         audio_file= open("speech.wav", "rb")
#         transcription = openai.Audio.transcribe("whisper-1", audio_file)['text']
#         conversation.append({'role': 'user', 'content': f'{transcription}'})
#         screen.addstr(4, 0, "You: {0}\n".format(transcription), curses.A_STANDOUT)
#         screen.addstr(6, 0, " . . . ", curses.A_NORMAL)
#         screen.refresh()

#         # Get ChatGPT response
#         response = openai.ChatCompletion.create(
#             model="gpt-3.5-turbo",
#             temperature=0.2,
#             messages=conversation
#         )

#         # Extract and display reply
#         reply = response['choices'][0]['message']['content']
#         conversation.append({'role': 'assistant', 'content': f'{reply}'})

#         # Split message from commands
#         split_reply = reply.split('##')
#         message = split_reply[0]
#         if len(split_reply) > 1:
#             command_string = split_reply[1]
#         else:
#             command_string = ""

#         # Indicate speaking (stop moving and blink)
#         ser.write(b'x')
#         time.sleep(0.05)
#         ser.write(b's')
#         time.sleep(0.05)

#         # Speak message
#         engine.say(message)
#         engine.runAndWait()
#         screen.addstr(8, 0, "NB3: {0}\n".format(message), curses.A_NORMAL)
#         screen.addstr(12, 0, "- commands: {0}\n".format(command_string), curses.A_STANDOUT)
#         screen.refresh()

#         # Indicate done speaking (stop moving and blink)
#         ser.write(b'x')
#         time.sleep(0.05)
#         ser.write(b'w')
#         time.sleep(0.05)

#         # Execute commands
#         commands = command_string.split(' ')
#         if(len(commands) > 1):
#             for c in commands:
#                 dir = c[0].encode('utf-8')
#                 dur = int(c[1:])
#                 dur_f = dur / 1000.0
#                 ser.write(dir)
#                 time.sleep(dur_f)

#         # Stop
#         ser.write(b'x')
#         time.sleep(0.05)

# finally:
#     # shut down
#     stream.close()
#     pya.terminate()
#     curses.nocbreak()
#     screen.keypad(0)
#     curses.echo()
#     curses.endwin()
#     ser.close()
# # FIN