import contextlib
import sys

import speech_recognition as sr
from gtts import gTTS
import os
import re

from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

MODEL_PATH = "/home/amory/llama.cpp/models/llama-2-7b.Q5_K_M.gguf"


def load_model() -> LlamaCpp:
    """Loads Llama model"""
    callback_manager: CallbackManager = CallbackManager([StreamingStdOutCallbackHandler()])
    # To have a nice display in the console

    llama_model: LlamaCpp = LlamaCpp(
        model_path=MODEL_PATH,
        temperature=0.5,
        max_tokens=2000,
        top_p=1,
        # callback_manager = callback_manager,
        verbose=True,
    )

    return llama_model


def load_model_gpu() -> LlamaCpp:
    """Loads Llama model with GPU"""
    callback_manager: CallbackManager = CallbackManager([StreamingStdOutCallbackHandler()])
    # To have a nice display in the console

    n_gpu_layers = 40
    n_batch = 1024  # number of Vram

    llama_model: LlamaCpp = LlamaCpp(
        model_path=MODEL_PATH,
        temperature=0.5,
        n_gpu_layers=n_gpu_layers,
        n_batch=n_batch,
        max_tokens=2000,
        top_p=1,
        # callback_manager = callback_manager,
        verbose=True,
    )

    return llama_model


@contextlib.contextmanager
def ignore_stderr():
    """Ignore annoying errors"""
    devnull = os.open(os.devnull, os.O_WRONLY)
    old_stderr = os.dup(2)
    sys.stderr.flush()
    os.dup2(devnull, 2)
    os.close(devnull)
    try:
        yield
    finally:
        os.dup2(old_stderr, 2)
        os.close(old_stderr)


def speak(text):
    """Speak the text with gTTs"""
    try:
        tts = gTTS(text=text, lang='fr')
        tts.save("temp.mp3")
        os.system("mpg321 -q temp.mp3")
        os.remove("temp.mp3")
    except AssertionError:
        pass


def listen(waiting):
    """Listen the user vocal entry
    waiting == True : waiting for the user to press Enter
    waiting == False : no waiting"""

    request = "start"
    with ignore_stderr():
        while request != "stop":
            # wait for input
            inp = "ajwvzhx"
            if waiting:
                while inp != "":
                    inp = input("Appuyez sur entrée pour parler\n")

            # audio reco
            r = sr.Recognizer()
            with sr.Microphone() as source:
                print("En attente d'une instruction...")
                audio = r.listen(source)

            try:
                request = r.recognize_google(audio, language='fr-FR')
                print(request)
                return request
            except sr.UnknownValueError:
                print("Je n'ai pas bien compris")
                speak("Je n'ai pas bien compris")
            except sr.RequestError as e:
                print("Could not request results from Google Speech Recognition service; {0}".format(e))
                request = "stop"


def findword(w):
    """Give None if the word is not in the specific String
    usage : findword("w")("Hello World !") --> something
    findword("a")("Hello World !") --> None
    so, findword("a")("Hello World !") is None --> True"""
    return re.compile(r'\b({0})\b'.format(w), flags=re.IGNORECASE).search


def clean_llm_answer(response):
    """Use to clean the response of the model"""
    answer = ""
    lines = response.split("\n")
    for line in lines:
        if findword("Answer")(line) is not None:
            try:
                answer = answer + re.search("Answer:(.*?)$", str(line)).group(1).rstrip()
                return answer
            except AttributeError:
                pass
        if findword("Comment")(line) is not None:
            if answer == "":
                try:
                    answer = answer + re.search("Comment:(.*?)$", str(line)).group(1).rstrip()
                except AttributeError:
                    pass
        if findword("Question")(line) is not None:
            pass
        if answer == "":
            answer = line

    return answer
