import speech_recognition as sr
import pyttsx3
import pyaudio
import time
import os
import sys

from langchain import PromptTemplate, LLMChain
from langchain.llms import GooglePalm


llm = GooglePalm(google_api_key="AIzaSyARIhyYDaD_vneSbeYApUHQPFLApXMsIcY", temperature=0.1)

template = """Task: Be a friend to a person. Talk about things that might calm down a person.
Topic: friendship, yoga, meditation
Style: Academic
Tone: Calm
Audience: 18-25 year olds
Length: 3 lines
Format: Text
Here's the question. {question}
"""

prompt = PromptTemplate(template=template, input_variables=["question"])
llm_chain = LLMChain(prompt=prompt, llm=llm, verbose=True)
llm_chain = LLMChain(prompt=prompt, llm=llm, verbose=True)

engine = pyttsx3.init()

def two(prompt):
    return (llm_chain(prompt)['text'])


def three(text):
    engine.say(text)
    engine.runAndWait()


def four():
    while True:
        print("start")
        with sr.Microphone() as source:
            r = sr.Recognizer()
            audio = r.listen(source)
            try:
                t = r.recognize_google(audio)
                print(t)
                if t:
                        print(t)
                        print(f"you said {t}")
                        res = two(t)
                        print(f"Virtual friend says {res}")
                        three(res)
                else:
                    three("I didn't Get it")

            except Exception as e:
                print("Error occured:{}".format(e))


def five():
    sys.exit("Stopped from talking")


if __name__ == "__main__":
    four()