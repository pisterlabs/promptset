import os
from dotenv import load_dotenv
from lib.speech_to_text import speech_to_text
from lib.speaking_python import SpeakingPython
import openai 

load_dotenv()  # take environment variables from .env.

class ChatGPTCommunication:
    URL = "https://api.openai.com/v1/completions"
    
    def __init__(self):
        self.AUTH_TOKEN = os.getenv("OPENAI_API_KEY")
    
    def text_completion(self, text: str):
        """
        Call ChatGPT API and return the response as json.
        API link: https://beta.openai.com/docs/api-reference/completions/create
        """
        openai.api_key = self.AUTH_TOKEN
        return openai.Completion.create(
            model="text-davinci-003",
            prompt=text,
            max_tokens=100,
            temperature=0
        )
    
    def _remove_first_and_last_newline_characters(self, text: str):
        while text.startswith('\n'):
            text = text[1:]
            
        while text.endswith('\n'):
            text = text[:-1]
        
        return text
    
    def _clean_text(self, text: str):
        """Returns a string with removed special characters."""
        cleaned_text = self._remove_first_and_last_newline_characters(text)
        
        # We're expecting mostly double or single '\n' characters
        cleaned_text = cleaned_text.replace("\n\n", '. ')
        cleaned_text = cleaned_text.replace("\n", '. ')
        
        return cleaned_text
    
    def ask(self, text: str):
        """Send some text to ChatGPT API and return the text response of AI"""
        json_response = self.text_completion(text)
        ai_response = json_response['choices'][0]['text']
        clean_response = self._clean_text(ai_response)
        return clean_response
    
    def voice_ask(self, language: str):
        """Ask a question using your microphone and hear back from chat gpt"""
        text = speech_to_text(language)
        response = self.ask(text)
        
        sp = SpeakingPython(language)
        sp.talk(response)
