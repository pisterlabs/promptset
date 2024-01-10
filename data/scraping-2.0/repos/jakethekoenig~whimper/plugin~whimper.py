import itertools
import os
import wave
from textwrap import dedent

import pyaudio
import pynvim
import openai
from faster_whisper import WhisperModel

openai.api_key = os.getenv("OPENAI_API_KEY")

AUDIO_FILE = "temp.wav"
CHUNK = 512
RATE = 44100
WHISPER_BATCH_SECS = 2
GPT_BATCH_SECS = 4
NUM_CHUNKS = 12000
audio = pyaudio.PyAudio()
MAX_SECONDS = 360 # TODO: break things up so I send at most 30 s to whisper

gpt_model = "gpt-3.5-turbo"
# TODO: I don't say the language but I should pick it up from context and let the model know in case it's from the beginning of a file.
system_prompt_content = """
You will be given a few lines of CODE and then a TRANSCRIPT of spoken code. Please
answer with just your best guess of the intended code continuation given the
transcript. Please only return the code continuation. The goal is for your response to
be inserted directly below the lines of CODE. Do not annotate or explain your code in
any way.
"""
system_prompt_content = dedent(system_prompt_content).strip()
system_prompt = {"role": "system",
                 "content": system_prompt_content}

def write_audio(data):
    waveFile = wave.open(AUDIO_FILE, 'wb')
    waveFile.setnchannels(1)
    waveFile.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b''.join(data))	
    waveFile.close()

def segments_to_transcript(segments):
    transcript = ""
    for segment in segments:
        transcript += segment.text + " "
    return transcript

@pynvim.plugin
class Whimper:
    def __init__(self, nvim):
        self.nvim = nvim

    def set_code_context(self):
        self.start_line = self.nvim.current.buffer.mark('.')[0]
        MAX_LINES = 20
        start_context_line = max(0, self.start_line - MAX_LINES)
        lines = self.nvim.current.buffer[start_context_line:self.start_line]
        self.code_context = "\n".join(lines)

    def gpt_prompt(self, transcript):
        history = [system_prompt]
        history += [{"role": "user", "content": "CODE:\n"+self.code_context}]
        history += [{"role": "user", "content": "TRANSCRIPT: "+transcript}]
        return history

    def setup(self):
        # TODO: replace 3 with selection menu
        # I keep a buffer that's much larger than the chunks I read so I don't lose frames when I call whisper and or GPT.
        self.stream = audio.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True, input_device_index=3, frames_per_buffer=CHUNK*50)
        # TODO: figure out CUDA
        self.whisper_model_size = "tiny"
        self.whisper_model = WhisperModel(self.whisper_model_size)
        self.initial_paste_value = self.nvim.command_output('set paste?')
        self.initial_stl_value = self.nvim.command_output('set stl?')
        self.nvim.feedkeys(':set paste\n')
        self.nvim.feedkeys(':let &stl="[Recording Audio]"\n')
        self.nvim.feedkeys("i")
        self.sent_so_far = ""
        self.set_code_context() # Don't need this for transcribe but it's cheap

    def teardown(self):
        self.stream.close()
        self.nvim.feedkeys("\x03")
        self.nvim.command('set {}'.format(self.initial_paste_value))
        self.nvim.command('set {}'.format(self.initial_stl_value))
        history = []

    def send_response(self, text):
        # TODO: detect typing and stop the model
        # It's possible to rewrite this with no ifs and it probably doesn't even matter for efficiency.
        if self.sent_so_far == text[:len(self.sent_so_far)]:
            self.nvim.feedkeys(text[len(self.sent_so_far):])
        else:
            current_line = self.nvim.current.buffer.mark('.')[0]
            if current_line == 1+self.start_line:
                self.nvim.feedkeys("\x03cc{}".format(text)) # TODO: multi line changes.
            else:
                self.nvim.feedkeys("\x03c{}k{}".format(current_line - self.start_line, text))
        self.sent_so_far = text

    @pynvim.function("Transcribe")
    def transcribe(self, args):
        self.setup()

        data = [] 
        last_whisper_run = 0
        for i in range(NUM_CHUNKS):
            if i * CHUNK / RATE > MAX_SECONDS:
                break
            data += [self.stream.read(CHUNK)]
            if (i - last_whisper_run) * CHUNK / RATE > WHISPER_BATCH_SECS:
                last_whisper_run = i
                # TODO: pass data directly to whisper
                write_audio(data)
                segments, info = self.whisper_model.transcribe(AUDIO_FILE, beam_size=5, language="en")
                transcript = segments_to_transcript(segments)
                if "stop" in transcript.lower():
                    break
                # TODO: detect typing and stop the model
                self.send_response(transcript)

        self.teardown()


    @pynvim.function("Whimper")
    def whimper(self, args):
        self.setup()

        data = [] 
        last_gpt_run = 0
        history = []
        for i in range(NUM_CHUNKS):
            if i * CHUNK / RATE > MAX_SECONDS:
                break

            data += [self.stream.read(CHUNK)]
            if (i - last_gpt_run) * CHUNK / RATE > GPT_BATCH_SECS:
                last_gpt_run = i
                # TODO: pass data directly to whisper
                write_audio(data)
                segments, info = self.whisper_model.transcribe(AUDIO_FILE, beam_size=5, language="en")
                transcript = segments_to_transcript(segments)
                if "stop" in transcript.lower():
                    break
                # TODO: remember previous GPT response and send it to GPT to minimize needed tokens.
                history = self.gpt_prompt(transcript)
                response = openai.ChatCompletion.create(
                    model=gpt_model,
                    messages=history
                    # stream=True
                )
                self.send_response(response["choices"][0]["message"]["content"])

        self.teardown()


