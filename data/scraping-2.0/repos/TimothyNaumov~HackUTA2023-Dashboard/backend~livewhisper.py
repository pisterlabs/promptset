import os
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
import openai
import json
from time import time as time2
from dotenv import load_dotenv

load_dotenv()

# This is my attempt to make psuedo-live transcription of speech using Whisper.
# Since my system can't use pyaudio, I'm using sounddevice instead.
# This terminal implementation can run standalone or imported for assistant.py
# by Nik Stromberg - nikorasu85@gmail.com - MIT 2022 - copilot

Model = "small"  # Whisper model size (tiny, base, small, medium, large)
English = True  # Use English-only model?
Translate = False  # Translate non-English to English?
SampleRate = 44100  # Stream device recording frequency
BlockSize = 100  # Block size in milliseconds
Threshold = 0.03  # Minimum volume threshold to activate listening
Lower_Threshold = 0.005  # Volume to not die
Vocals = [1, 2000]  # Frequency range to detect sounds that could be speech
EndBlocks = 20  # Number of blocks to wait before sending to Whisper

openai.api_key = os.environ["OPENAI_API_KEY"]


class StreamHandler:
    def __init__(self, assist=None):
        if (
            assist == None
        ):  # If not being run by my assistant, just run as terminal transcriber.

            class fakeAsst:
                running, talking, analyze = True, False, None

            self.asst = fakeAsst()  # anyone know a better way to do this?
        else:
            self.asst = assist
        self.running = True
        self.padding = 0
        self.prevblock = self.buffer = np.zeros((0, 1))
        self.fileready = False
        self.time_limit = time2() + 30

    def callback(self, indata, frames, time, status):
        # if status: print(status) # for debugging, prints stream errors.
        if not any(indata):
            print(
                "\033[31m.\033[0m", end="", flush=True
            )  # if no input, prints red dots
            # print("\033[31mNo input or device is muted.\033[0m") #old way
            # self.running = False  # used to terminate if no input
            return
        # A few alternative methods exist for detecting speech.. #indata.max() > Threshold
        # zero_crossing_rate = np.sum(np.abs(np.diff(np.sign(indata)))) / (2 * indata.shape[0]) # threshold 20
        freq = np.argmax(np.abs(np.fft.rfft(indata[:, 0]))) * SampleRate / frames
        # print("Threshold is ", np.sqrt(np.mean(indata**2)))
        if np.sqrt(np.mean(indata**2)) > Lower_Threshold:
            self.time_limit = time2() + 30
        elif self.time_limit < time2():
            self.running = False
            print("\033[31mNo input or device is muted.\033[0m")
            return
        if (
            np.sqrt(np.mean(indata**2)) > Threshold
            and Vocals[0] <= freq <= Vocals[1]
            and not self.asst.talking
        ):
            print(".", end="", flush=True)
            if self.padding < 1:
                self.buffer = self.prevblock.copy()
            self.buffer = np.concatenate((self.buffer, indata))
            self.padding = EndBlocks
        else:
            self.padding -= 1
            if self.padding > 1:
                self.buffer = np.concatenate((self.buffer, indata))
            elif (
                self.padding < 1 < self.buffer.shape[0] > SampleRate
            ):  # if enough silence has passed, write to file.
                self.fileready = True
                write(
                    "dictate.wav", SampleRate, self.buffer
                )  # I'd rather send data to Whisper directly..
                self.buffer = np.zeros((0, 1))
            elif (
                self.padding < 1 < self.buffer.shape[0] < SampleRate
            ):  # if recording not long enough, reset buffer.
                self.buffer = np.zeros((0, 1))
                print("\033[2K\033[0G", end="", flush=True)
            else:
                self.prevblock = (
                    indata.copy()
                )  # np.concatenate((self.prevblock[-int(SampleRate/10):], indata)) # SLOW

    def process(self):
        if self.fileready:
            print("\n\033[90mTranscribing..\033[0m")
            result = ""
            with open("dictate.wav", "rb") as f:
                result = openai.Audio.transcribe("whisper-1", f)
            # result = self.model.transcribe(
            #     "dictate.wav",
            #     fp16=False,
            #     language="en" if English else "",
            #     task="translate" if Translate else "transcribe",
            # )
            print(f"\033[1A\033[2K\033[0G{result['text']}")
            if self.asst.analyze != None:
                self.asst.analyze(result["text"])
            self.fileready = False

    def listen(self):
        print("\033[32mListening.. \033[37m(Ctrl+C to Quit)\033[0m")
        with sd.InputStream(
            channels=1,
            callback=self.callback,
            blocksize=int(SampleRate * BlockSize / 1000),
            samplerate=SampleRate,
            device=13,
        ):
            while self.running and self.asst.running:
                self.process()


def main():
    try:
        handler = StreamHandler()
        handler.listen()
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        print("\n\033[93mQuitting..\033[0m")
        if os.path.exists("dictate.wav"):
            os.remove("dictate.wav")


if __name__ == "__main__":
    main()  # by Nik
