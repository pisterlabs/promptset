# import openai
# import os
# from controllers.translation_controller import TranslationController
# import streamlit as st


# if "open_ai_api_key.txt" in os.listdir():
#     with open("open_ai_api_key.txt", "r") as f:
#         api_key = f.read()
# else:
#     api_key = st.text_input("OpenAI API key", type="password", key="open_ai_api_key")

# openai.api_key = api_key

# class AiTrancriber():

#     def __init__(self, audio_file_name, target_language, prompt_text=f"""
#         Break down the following transcript into short phrases and sentences. Each phrase should be on a new line.      
#         transcript:        
#         """):        
#         self.audio_file_name = audio_file_name
#         self.target_language = target_language
#         self.prompt_text = prompt_text
#         self.transcription_model = "whisper-1"
#         self.model_engine = "gpt-3.5-turbo"
#         self.transcript = None
#         self.audio_file = None        
    
#     def open_audio_file(self):
#         print("audio_file_name", self.audio_file_name)
#         self.audio_file = open(self.audio_file_name, "rb")
#         print("audio_file", self.audio_file)
#         return self.audio_file

#     def transcribe(self):
#         self.audio_file = self.open_audio_file()        
#         transcript = openai.Audio.transcribe(self.transcription_model, self.audio_file)    
#         return transcript
    
#     def format_phrases(self):
#         self.transcript = self.transcribe()
#         prompt = [{"role": "user", "content": f'Break down the following transcript into short phrases and sentences. Each phrase should be on a new line: "{self.transcript.text  }"'}]
#         # prompt = self.prompt_text + self.transcript.text        

#         important_phrases_response = openai.ChatCompletion.create(
#             model=self.model_engine,
#             # prompt=prompt,
#             messages = prompt
#         )
#         ai_generated_phrases = important_phrases_response.choices[0].message.content    
#         # remove all empty lines
#         ai_generated_phrases = os.linesep.join([s for s in ai_generated_phrases.splitlines() if s])
#         # remove the following characters from the beginning of each line ['"','-', ' ']
#         ai_generated_phrases = os.linesep.join([s[1:] if s[0] == '"' else s for s in ai_generated_phrases.splitlines()])
#         # add a translation to the phrases using TranslationController. Add the translation to the end of each line, separated by a semicolon
#         translation_controller = TranslationController(ai_generated_phrases, self.target_language)
#         ai_generated_phrases = translation_controller.multi_line()
#         return ai_generated_phrases        