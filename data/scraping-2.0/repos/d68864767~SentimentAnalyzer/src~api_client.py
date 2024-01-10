# src/api_client.py

import openai
import re

class OpenAIApiClient:
    def __init__(self):
        self.api_key = 'your_openai_api_key'  # Replace with your OpenAI API key
        openai.api_key = self.api_key

    def preprocess_text(self, text):
        # Remove URLs
        text = re.sub(r'http\S+|www.\S+', '', text)

        # Remove non-alphabetic characters
        text = re.sub(r'[^a-zA-Z\s]', '', text)

        # Convert to lowercase
        text = text.lower()

        return text

    def generate_text(self, prompt, max_tokens=100):
        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=prompt,
            max_tokens=max_tokens
        )

        return response.choices[0].text.strip()
