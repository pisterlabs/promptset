import requests
import json
from PyPDF2 import PdfReader
import openai
import os

#pdf content should not be greater than 4096 tokens


def extract_text(pdf_path):

    with open(pdf_path, 'rb') as pdf_file:
            # Create a PDF reader object
            pdf_reader = PdfReader(pdf_file)
            text=''
            for page in pdf_reader.pages:
                text+=page.extract_text()
    return text

class CustomError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(message)

def api_status(key):
    openai.api_key=key
    # Try to create a completion
    try:
        response = openai.Completion.create(
            engine="text-davinci-001",
            prompt="What is the meaning of life?",
            temperature=0.5,
            max_tokens=60,
            top_p=0.3,
            frequency_penalty=0.5,
            presence_penalty=0.0,
        )
    except openai.OpenAIError as e:
        return False
    else:

       
        return True

def get_chat_response(question,api_key):

    if not api_status(api_key):
        raise CustomError('api key is not valid')
    # API endpoint
    url = 'https://api.openai.com/v1/chat/completions'

    # Your OpenAI API key
    api_key = api_key

    # Request headers
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }

    # Request payload
    payload = {
        'model': 'gpt-3.5-turbo',
        'messages': [{'role': 'system', 'content': 'You are a helpful assistant.'},
                     {'role': 'user', 'content': question}]
    }

    response = requests.post(url, headers=headers, json=payload)


    data = response.json()

    
    reply= data['choices'][0]['message']['content']

    return reply

def gpt_extract_info(pdf_path):
# Example usage
    user_question = input("Ask a question: ")
    response = get_chat_response(extract_text(pdf_path) + '\n\n'+user_question)
    return response


