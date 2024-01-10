import os
import openai
from dotenv import load_dotenv
class AIService:
    def __init__(self):
        # Load environment variables from .env file
        load_dotenv()

        # Get the API key from the environment variable
        api_key = os.getenv("OPENAI_API_KEY")

        # Set the API key for OpenAI
        openai.api_key = api_key
        
    def ask(self, prompt, document):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f'{document}'},
                {"role": "user", "content": f'{prompt}'}
            ]
        )
        return response['choices'][0]['message']['content']

