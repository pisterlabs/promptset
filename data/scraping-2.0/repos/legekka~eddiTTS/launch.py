import os
import time
import json
import openai
import keyboard
from modules.Apis import TGWapi, TTSapi
from scipy.io.wavfile import write
from modules.Gui import Gui
from PySide6.QtWidgets import QApplication
import sys
import re
import numpy as np

speechresponderpath = os.path.join(os.getenv("APPDATA"), "EDDI", "speechresponder.out")
linescontent = []
lineslength = 0


def init_whisper():
    import whisper

    global whisper_model
    whisper_model = whisper.load_model("base")


def load_config():
    global config
    with open("config.json", "r") as f:
        config = json.load(f)

    # load prompts
    with open(config["prompts"]["ask"]["alpaca"], "r") as f:
        config["prompts"]["ask"]["alpaca"] = f.read()

    with open(config["prompts"]["rephrase"]["alpaca"], "r") as f:
        config["prompts"]["rephrase"]["alpaca"] = f.read()

    with open(config["prompts"]["rephrase2"]["alpaca"], "r") as f:
        config["prompts"]["rephrase2"]["alpaca"] = f.read()

    with open(config["prompts"]["ask"]["vicuna"], "r") as f:
        config["prompts"]["ask"]["vicuna"] = f.read()

    with open(config["prompts"]["rephrase"]["vicuna"], "r") as f:
        config["prompts"]["rephrase"]["vicuna"] = f.read()

    with open(config["prompts"]["rephrase2"]["vicuna"], "r") as f:
        config["prompts"]["rephrase2"]["vicuna"] = f.read()

    with open(config["prompts"]["ask"]["openai"], "r") as f:
        config["prompts"]["ask"]["openai"] = f.read()

    with open(config["prompts"]["rephrase"]["openai"], "r") as f:
        config["prompts"]["rephrase"]["openai"] = f.read()
    


def load_messages():
    global messages
    with open("data/messages.json", "r") as f:
        messages = json.load(f)


def ask_text_OpenAI(text):
    system_prompt = config["prompts"]["ask"]["openai"]
    start = time.time()
    print("OpenAI API time: ", end="", flush=True)
    last_messages = messages[-30:]
    last_messages = list(
        map(
            lambda message: {"role": message["role"], "content": message["text"]},
            last_messages,
        )
    )
    composed_prompt = (
        [{"role": "system", "content": system_prompt}]
        + last_messages
        + [{"role": "user", "content": text}]
    )

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", messages=composed_prompt
    )

    print(str(round(time.time() - start, 2)) + "s")
    return response.choices[0].message.content


def rephrase_text_OpenAI(text):
    system_prompt = config["prompts"]["rephrase"]["openai"]
    start = time.time()
    print("OpenAI API time: ", end="", flush=True)
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
        )
    except:
        print("OpenAI API error, using original text")
        return text
    print(str(round(time.time() - start, 2)) + "s")
    return response.choices[0].message.content


def init_openai_api():
    openai.api_key = config["openai_api_key"]


def logText(text, role="assistant"):
    message = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
        "role": role,
        "text": text.strip(),
    }
    messages.append(message)
    with open("data/messages.json", "w") as f:
        json.dump(messages, f, indent=4)

def audioLength(audio, sr):
    # calculates the length of an audio in milliseconds
    return int(len(audio) / sr * 1000)

def audiosLength(audios, sr):
    # calculates the length of a list of audios in milliseconds
    return sum([audioLength(audio, sr) for audio in audios])

def playQueue(audios, sr):
        # concatenate every audio in one audio
        audio = audios[0]
        for i in range(1, len(audios)):
            audio = np.concatenate((audio, audios[i]))
        # play the audio
        ttsapi.just_play(audio, sr)

def checkPunctuation(text):
    if text[-1] not in [".", "!", "?"]:
        text += "."
    return text

def splitAndPlaySentences(text, debug=False):
    # splitting text into sentences
    texts = re.split(r"([.!?]) ", text)
    # putting back the punctuation
    texts = ["".join(texts[i : i + 2]) for i in range(0, len(texts), 2)]
    if len(texts) > 1:
        audios = []
        sr = 0
        for txt in texts:
            txt = txt.strip()
            txt = checkPunctuation(txt)
            if len(txt) > 0:
                audio, sr = ttsapi.generate(txt, play=False, debug=debug)
                audios.append(audio)
        if config["gui"]:
            gui.display_message(text, audiosLength(audios, sr) + 2000)
        print("Speaking: " + text)
        playQueue(audios, sr)
        if config["gui"]:
            gui.wait()
    else:
        audio, sr = ttsapi.generate(text, play=False, debug=debug)
        if config["gui"]:
            gui.display_message(text, audioLength(audio, sr) + 2000)
        print("Speaking: " + text)
        ttsapi.just_play(audio, sr)
        if config["gui"]:
            gui.wait()

def playSentences(text, debug=False):
    # we don't split the text, just create the audio and play it
    audio, sr = ttsapi.generate(text, play=False, debug=debug)
    if config["gui"]:
        gui.display_message(text, audioLength(audio, sr) + 2000)
    print("Speaking: " + text)
    ttsapi.just_play(audio, sr)
    if config["gui"]:
        gui.wait()
    
# loop of the checker and speaker
def checkForChangesAndSpeak():
    global lineslength
    try:
        with open(speechresponderpath, "r") as f:
            lines = f.readlines()
            if len(lines) > lineslength:
                print("New line detected")

    except:
        print("Could not read file")
        return

    if lineslength == 0:
        lineslength = len(lines) - 1

    if (len(lines) > 0) and lineslength != len(lines):
        i = lineslength
        while i < len(lines):
            start = time.time()
            text = lines[i]
            # get last 3 lines from the current line
            logs = lines[i - 3 : i]
            # strip the lines
            logs = [log.strip() for log in logs]
            #text = TGWapi(config).rephrase2(eddi_message=lines[i], logs=logs, debug=True, max_new_tokens=150)
            text = TGWapi(config).rephrase(text, debug=True, max_new_tokens=150)
            logText(text, role="assistant")
            splitAndPlaySentences(text, debug=True)
            print("Total time: " + str(round(time.time() - start, 2)) + "s")
            i += 1
        lineslength = len(lines)


def checkForKeypress():
    if keyboard.is_pressed("ctrl+*"):
        question = input("Question: ")
        answer = TGWapi(config).ask(
            question, context_messages=messages[-30:], max_new_tokens=400, debug=True
        )
        logText(question, role="user")
        logText(answer, role="assistant")
        playSentences(answer, debug=True)

    if config["whisper"]["use"]:
        if keyboard.is_pressed("ctrl+y"):
            ttsapi.generate("Yes, Commander?", play=True, debug=True)
            audio = ttsapi.record()

            write("tmp/recorded.wav", 44100, audio)
            print("Transcribing...")
            result = whisper_model.transcribe("tmp/recorded.wav", language="english")
            question = result["text"].strip()
            print("Recorded text: " + question)
            answer = TGWapi(config).ask(
                question,
                context_messages=messages[-30:],
                max_new_tokens=400,
                debug=True,
            )
            logText(question, role="user")
            logText(answer, role="assistant")
            playSentences(answer, debug=True)


def init_gui():
    global app
    global gui

    app = QApplication(sys.argv)
    gui = Gui()
    gui.show()


def main():
    load_config()
    load_messages()
    if config["whisper"]["use"]:
        init_whisper()
    init_openai_api()

    global ttsapi
    ttsapi = TTSapi(config)

    if config["gui"]:
        init_gui()
    print("test")

    while True:
        checkForChangesAndSpeak()
        checkForKeypress()
        if config["gui"]:
            # when gui is used, we will use the gui loop for waiting
            gui.sleep(250)
        else:
            time.sleep(0.25)
            
    # this is only needed if we don't use the loop
    if config["gui"]:
        sys.exit(app.exec_())

if __name__ == "__main__":
    main()
