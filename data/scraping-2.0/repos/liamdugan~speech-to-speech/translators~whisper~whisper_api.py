import openai, os

class WhisperAPITranslator:
    def __init__(self, api_key, organization):
        openai.api_key = api_key
        openai.organization = organization
    
    def translate(self, file, prompt):
        audio_file = open(file, "rb")
        return openai.Audio.translate("whisper-1", audio_file, 
                                      language="en", prompt=prompt,
                                      response_format="verbose_json")