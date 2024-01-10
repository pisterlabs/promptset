from transformers import pipeline
from PyPDF2 import PdfReader
import os
from tqdm import tqdm
from transformers import AutoTokenizer
import math
import openai
from bs4 import BeautifulSoup
import pandas as pd
import requests
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from dotenv import load_dotenv

load_dotenv()

openai.api_key = "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"

uri = "mongodb+srv://admin:admin@casecluster.smlo0.mongodb.net/?retryWrites=true&w=majority"

client = MongoClient(uri, server_api = ServerApi('1'))

hf_name = 'pszemraj/led-large-book-summary'

tokenizer = AutoTokenizer.from_pretrained(hf_name)

summarizer = pipeline(
    "summarization",
    hf_name,
    device = 0
)

summaries = []

html  = ''
with open('2022.html', 'r') as r:
    html = r.read()

soup = BeautifulSoup(html, 'html.parser')
index = 1018

for element in tqdm(soup.find_all('tr')[1+index:]):
    case_number = element.find('td').text
    party_involved = element.find_all('td')[1].text
    href = element.find('a')['href']
    response = requests.get(str(requests.get(href).content).split("\\\'")[1])
    with open(f"{party_involved.replace('/', '-')}.pdf", 'wb') as pdf_file:
        pdf_file.write(response.content)
    prompt = ''
    try:
        reader = PdfReader(f"{party_involved.replace('/', '-')}.pdf")
    except:
        client.Case.BHC.insert_one({'index' : index, 'case_number' : case_number, 'case_name' : party_involved, 'case_text' : 'error', 'case_description' : 'error'})
        continue
    os.remove(f"{party_involved.replace('/', '-')}.pdf")
    for n, page in enumerate(reader.pages):
        prompt += page.extract_text() + '\n'
    case_text = prompt
    inputs = tokenizer.encode_plus(prompt, add_special_tokens = False, return_tensors = "pt")
    num_tokens = inputs["input_ids"].shape[-1]
    if num_tokens < 16384:
        result = summarizer(prompt, min_length = 16, max_length = 256, no_repeat_ngram_size = 3, encoder_no_repeat_ngram_size = 3, repetition_penalty = 3.5, num_beams = 4, early_stopping = True)[0]['summary_text']
    else:
        divisions = math.ceil(num_tokens / 16384) + 1
        prompt = ''
        summary_gpt = ''
        k = 0
        for n, page in enumerate(reader.pages):
            prompt += page.extract_text() + '\n'
            if k == int(len(reader.pages) / divisions):
                result = summarizer(prompt, min_length = 16, max_length = 256, no_repeat_ngram_size = 3, encoder_no_repeat_ngram_size = 3, repetition_penalty = 3.5, num_beams = 4, early_stopping = True)[0]['summary_text']
                prompt = ''
                summary_gpt += '\n\n' + result
            k += 1
        response = openai.ChatCompletion.create(
                model = "gpt-3.5-turbo",
                messages=[
                        {"role": "system", "content": "You are a helpful assistant who answers questions given a tabular data."},
                        {"role": "user", "content": f"Below are part by part summary of a case judgement beacause it couldn't be summarised as one. Combine the summaries and ensure that you only use the context provided within the summaries below. The final answer should be within 256 words and in the style of the summaries provided to you. {summary_gpt}"},
                    ]
                )
        prompt = summary_gpt
        result = response['choices'][0]['message']['content'].replace('\n', ' ')
    index += 1
    client.Case.BHC.insert_one({'index' : index, 'case_number' : case_number, 'case_name' : party_involved, 'case_text' : prompt, 'case_description' : result})