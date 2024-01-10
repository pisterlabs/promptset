import os
import json
import pandas as pd
import numpy as np
from transformers import AutoTokenizer
import openai
from dotenv import load_dotenv
from utils import api_config
from glob import glob

txt_list = glob('/home/jaeyoung/xai_ocr/pythonProject/a/pre2(png)/**/*.txt', recursive=True)
print(len(txt_list))

class InformationExtractAPI:
    def __init__(self, txt):
        self.txt = txt

    def fewshot_context(self):
        instruction = """
        Read the following OCR-extracted transport logistics document and extract the departure and destination locations.
        """
      
        # prompt = instruction + "\n" + answer_format + "\n context:" + self.txt
        prompt = instruction + "\n" + "OCR context: \n" + self.txt
        # prompt = instruction + "\n OCR context: \n" + self.txt

        return prompt

def openai_request(prompt):
    completion = openai.ChatCompletion.create(
        model = openai_config['model'],
        messages=[{"role": "user", "content": prompt}],
        temperature = openai_config['temperature'],
        top_p = openai_config['top_p'],
        max_tokens = openai_config['max_tokens'],
        presence_penalty = openai_config['presence_penalty'],
        frequency_penalty = openai_config['frequency_penalty'],
    )

    print(completion)

if __name__ == "__main__":

    txt_file = txt_list[0]
    with open(txt_file, 'r', encoding='utf-8') as f:
        txt = f.read()

    openai.api_key = open("openai_key.txt", "r").read().strip("\n")
    openai_config = api_config.openai_config
    
    info_ext = InformationExtractAPI(txt)
    print(info_ext.fewshot_context())
    ex_prompt = info_ext.fewshot_context()
    openai_request(ex_prompt)

