import os
import openai
import sys


class Translator():

    def __init__(self):
        openai.api_key = "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"
        self.api_key = openai.api_key
        
        # Set up dictionary of language codes
        self.ENGLISH = 'en'
        self.ICELANDIC = 'is'
        self.available_languages = {self.ENGLISH, self.ICELANDIC}


    def build_command(self, lang, sentence):
        if lang == self.ENGLISH:
            return "Translate this into 1. Icelandic:\n\n" + sentence + "\n\n1."
        elif lang == self.ICELANDIC:
            return "Translate this into 1. English:\n\n" + sentence + "\n\n1."
        else:
            sys.exit(f'Translator: translation from {lang} unsupported.\n')


    def translate(self, from_lang, to_lang, sentence):
        # Ensure input type
        if not isinstance(sentence, str):
            sys.exit('Translator: Given sentence is not a string.\n')

        command = self.build_command(from_lang, sentence)
        response = openai.Completion.create(
            model="text-davinci-002",
            prompt=command,
            temperature=0.3,
            max_tokens=100,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )

        return (response.choices[0].text).replace('\n', '')
