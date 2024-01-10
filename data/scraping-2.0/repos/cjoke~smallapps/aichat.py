#!/usr/bin/env python

import openai
import time
from utils import microphone

api_key = $OPENAI_API_KEY
openai.api_key = api_key

mypromt = [
    {"role": "system", "content": "You are a nice intelligent python code assistant."}
]


def log_conversation(conversation, filename):
    with open(filename, "a") as logfile:
        logfile.write(f"\nTIMESTAMP: {time.ctime()}\n")
        for line in conversation:
            logfile.write(line)


class TextFormatter:
    def __init__(self, text):
        self.text = text

    def format_output(self):
        chunks = [self.text[i : i + 100] for i in range(0, len(self.text), 100)]
        formatted_text = "\n".join(chunks)
        return formatted_text


while True:
    recording = microphone.SpeechRecognizer()
    recorded_message = recording.recognize_speech()

    if recorded_message == "text please":
        recorded_message = input(" Message to chatGPT here : ")

    if recorded_message == "exit":
        exit()

    if recorded_message:
        mypromt.append(
            {"role": "user", "content": recorded_message},
        )
        mycontent = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=mypromt,
            temperature=0.7,
            max_tokens=1024
        )

    reply = mycontent.choices[0].message.content
    formatter = TextFormatter(reply)
    formatted_output = formatter.format_output()
    reply = formatted_output
    logger = f"USER INPUT : {recorded_message} " + "\n" + f" ChatGPT :{reply} "
    print(f"CHAT GPT: {reply}")
    mypromt.append({"role": "assistant", "content": reply})
    log_conversation(logger, "conversation_log.txt")
