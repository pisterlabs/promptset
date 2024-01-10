from OpenAIUtil.Operations import *
import torch
from transformers import pipeline
import openai

class TranscribeOperations(Operations):
    
    def __init__(self):
        self.MODEL_NAME = "openai/whisper-large-v2"
        self.device = 0 if torch.cuda.is_available() else "cpu"

    def setOpenAIConfig(self, api_key: str, org_id: str):
        if org_id is not None:
            openai.organization = org_id
        openai.api_key = api_key
        
    def initialize(self):
        self.pipe = pipeline(
            task="automatic-speech-recognition",
            model=self.MODEL_NAME,
            chunk_length_s=30,
            device=self.device,
        )

        self.all_special_ids = self.pipe.tokenizer.all_special_ids
        self.transcribe_token_id = self.all_special_ids[-5]
        self.translate_token_id = self.all_special_ids[-6]


    
    def transcribe(self, audio_file: str, language: str ="en"):
        try: 
            if audio_file is not None and openai.api_key is not None:
                audio = open(audio_file, "rb")
                transcript = openai.Audio.transcribe("whisper-1", audio, language = language)
                return transcript["text"], transcript["text"]
            else:
                return "", ""
        except openai.error.OpenAIError as error_except:
            print("TranscribeOperations transcribe")
            print(error_except.http_status)
            print(error_except.error)
            return error_except.error["message"], ""


    def transcribe_whisper_large_v2(self, audio_file, task="transcribe"):
        self.initialize()
        self.pipe.model.config.forced_decoder_ids = [[2, self.transcribe_token_id if task=="transcribe" else self.translate_token_id]]
        text = self.pipe(audio_file)["text"]
        return text, text



