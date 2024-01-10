# imports
from openai import OpenAI
from dotenv import load_dotenv
import os

class Assistant:
    def __init__(self, name, personality):
        self.name = name
        self.personality = personality
        self.session_id = None

    def start_session(self):
        load_dotenv() # Load the environment
        api_key = os.getenv('OPENAI_API_KEY')
        client = OpenAI()

        self.session_id = client.beta.assistants.create(model="gpt-4").id

    def generate_response(self, message):
        if not self.session_id:
            self.start_session()

        response = openai.Message.create(
            assistant_id=self.session_id,
            messages=[
                {"role": "system", "content": self.personality},
                {"role": "user", "content": message}
            ]
        )
        return response.data[0]['content']