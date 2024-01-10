import openai
import os
# from controllers.translation_controller import TranslationController
from services.translator import Translator
import streamlit as st


class AiFormatter():

    def __init__(self, transcript, api_key):
        self.transcript = transcript
        self.api_key = api_key
        self.model_engine = "gpt-3.5-turbo"
    
    def format_phrases(self):
        prompt = [{"role": "user", "content": f'Break down the following transcript into short phrases and sentences. Each phrase should be on a new line: "{ self.transcript }"'}]
        # prompt = self.prompt_text + self.transcript.text        
        openai.api_key = self.api_key
        important_phrases_response = openai.ChatCompletion.create(
            model=self.model_engine,
            # prompt=prompt,
            messages = prompt
        )
        ai_generated_phrases = important_phrases_response.choices[0].message.content    
        # remove all empty lines
        ai_generated_phrases = os.linesep.join([s for s in ai_generated_phrases.splitlines() if s])
        # remove the following characters from the beginning of each line ['"','-', ' ']
        ai_generated_phrases = os.linesep.join([s[1:] if s[0] == '"' else s for s in ai_generated_phrases.splitlines()])
        # add a translation to the phrases using TranslationController. Add the translation to the end of each line, separated by a semicolon

        # translation_controller = Translator.multi_line(ai_generated_phrases, self.target_language)
        # ai_generated_phrases = translation_controller.multi_line()
        return ai_generated_phrases  

                