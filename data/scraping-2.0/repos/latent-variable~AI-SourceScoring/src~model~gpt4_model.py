# gpt4_model.py
import os
from openai import OpenAI


# Function to read API key from a file
def read_api_key(file_path = r'./api_key.txt'):
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            return file.read().strip()
    else:
        return None
    

class GPT4Model:
    def __init__(self, api_key):
        
        self.client = OpenAI(api_key = read_api_key())  
        self.model = 'gpt-4-1106-preview'
        self.history = []

    def generate_response(self, prompt, max_tokens=100):
        try:
            self.history.append({"role": "user", "content": [{"type": "text", "text": prompt}]})
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.history,
                max_tokens=max_tokens
            )
            llm_response = response.choices[0].message.content.strip()  # Extract and clean the output text
            self.history.append({"role": "assistant", "content": [{"type": "text", "text": llm_response}]})
            print(llm_response)


        except Exception as e:
            print(e)
            llm_response = "No reponse"
        return llm_response
