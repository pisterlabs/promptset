
from dotenv import load_dotenv
from openai import OpenAI
import os
import sys

load_dotenv()

from schema.request import CreateStarRequest
from ai_models.text_generation.characteristic_generation import merge_prompt_text,get_characteristics 
from ai_models.text_generation.chat_generation import insert_persona_to_prompt, merge_prompt_input, prepare_chat

from ai_models.speaker_identification.clova_speech import ClovaSpeechClient
from ai_models.speaker_identification.postprocessing import speaker_diarization

import json
from io import BytesIO
from pydub import AudioSegment

class TextGeneration:
    API_KEY = os.getenv("GPT_API_KEY")
    client = OpenAI(api_key=API_KEY)

    def __init__(self, request:CreateStarRequest) -> None:
        
        self.original_text_file = request.original_text_file
        self.star_gender = request.gender
        self.star_name = request.star_name
        self.persona = request.persona
        self.relationship = request.relationship

        self.API_KEY = os.getenv("GPT_API_KEY")
        self.client = OpenAI(api_key=self.API_KEY)

        # 추후 수정
        self.prompt_file_path = 'prompt_data/extract_characteristic.txt'
        self.system_input_path = "prompt_data/system_input.txt"

    def create_prompt_input(self) -> str:

        # 텍스트 파일 열기


        #카카오톡 텍스트 원본에서 고인의 텍스트만 잘라내기
        star_text = "" #고인의 텍스트만 추출한 것


        # process for extracting characteristics
        prompt = merge_prompt_text(star_text,self.prompt_file_path)
        characteristics = get_characteristics(prompt,self.client)
        
        # process for preparing system prompt
        system_input = insert_persona_to_prompt(self.star_name,self.relationship,self.system_input_path)
        chat_prompt_input_data = merge_prompt_input(characteristics,system_input,star_text)
        
        return chat_prompt_input_data



    
class SpeakerIdentification:

    def __init__(self,original_voice_file):
        self.original_voice_file = original_voice_file
        self.speech_list = []

    def get_speaker_samples(self):
        audio_byte = BytesIO(self.original_voice_file.file.read())
        audio_seg = AudioSegment.from_file(audio_byte)
        audio_binary = audio_seg.export(format="wav").read()
        res = ClovaSpeechClient().req_upload(file=audio_binary, completion='sync')
        timestamp = json.loads(res.text)

        speaker_num, speech_list, speaker_sample_list = speaker_diarization(timestamp)
        self.speech_list = speech_list

        # speaker_num: speaker 수, speaker_sample_list: speaker 각자의 목소리 담긴 리스트
        # 이것들을 프론트에 넘겨줄 수 있도록 작업


        return None

    def get_star_voice(self):
        # speech_list 가져와서 고인 목소리 이어붙이는 작업

        pass

    

