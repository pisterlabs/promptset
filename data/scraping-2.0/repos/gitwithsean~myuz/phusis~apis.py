import openai
from termcolor import colored
from .phusis_utils import retry_with_exponential_backoff


class OpenAiAPI():
    api = openai
    
    def __init__(self):
        with open('./.secrets/openai_api_key', 'r') as f:
            openai.api_key = f.read()
            
    
    @retry_with_exponential_backoff(max_retries=5, initial_delay=0.1)   
    def chat_response(self, api_data):
                
        # print(colored("\nOpenAiApi.chat_response(): Submitting chat completion\n", "yellow"))
        
        completion = openai.ChatCompletion.create(
            model=api_data['model'],
            messages=api_data['messages_to_submit']
        )
        
        # print(colored("\nOpenAiApi.chat_response(): Response received\n", "green"))
        # print(colored(f"\n{completion}\n", "green"))
        return completion

    @retry_with_exponential_backoff(max_retries=5, initial_delay=0.1) 
    def get_embeddings_for(self, input, model="text-embedding-ada-002"):
        response = openai.Embedding.create(
            model=model,
            input=input,
        )
        return response['data'][0]['embedding']