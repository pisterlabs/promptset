#!/usr/bin/python

import configparser
import json
import logging

from openai import OpenAI

from core.alive import alive
from core.emailx import Emailx
from core.twitterc import TwitterC
from core.voicerecognition import VoiceRecognition

class QueryMaster(object):

    def __init__(self, voicesynthetizer):

        self.modulename = 'VoiceExperimental'
        self.voicesynthetizer = voicesynthetizer
        self.emailx = Emailx()
        self.twitterc = TwitterC('twython')
        self.voicerecognition = VoiceRecognition(self.voicesynthetizer)

        self.setup()

    def __del__(self):
        pass

    def setup(self):

        logging.info('Voice Experimental Setup')
        self.voicerecognition.languageset('spanish')
        self.voicesynthetizer._set_language_argument("spanish")

    def cleanup(self):

        logging.info('Voice Experimental Cleanup')
        self.voicerecognition.languageset('spanish')
        self.voicesynthetizer._set_language_argument("spanish")

    def query(self, message):

        services = configparser.ConfigParser()
        path = "configuration/services.config"
        services.read(path)
        api_key = services.get("openai", "api_key")
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": message,
                }
            ],
            model="gpt-3.5-turbo-1106",
        )
        return(response.model_dump()['choices'][0]['message']['content'])


    def listen(self):

        logging.info('Voice Experimental Listen')
        self.voicesynthetizer.speech_it('Hola! Cual es tu pregunta?')
        self.voicerecognition.record()
        question = self.voicerecognition.recognize('False')
        answer = self.query(question)
        self.voicesynthetizer.speech_it(answer)

# End of File
