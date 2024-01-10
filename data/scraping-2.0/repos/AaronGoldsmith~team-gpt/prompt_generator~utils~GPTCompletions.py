import openai
import os
from dotenv import load_dotenv

class GPTCompletions:

    def __init__(self):
        # Load API key from environment variables
        load_dotenv(os.path.join(os.path.dirname(__file__), '..','..', '.env'))
        openai.api_key = os.getenv("OPENAI_API_KEY")

    def gpt_response(self, prompt):
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return completion.choices[0]
