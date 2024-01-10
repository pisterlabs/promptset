import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

class LanguageModelService:
    def __init__(self, provider='openai'):
        self.provider = provider
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OpenAI API key not found.")
        
        self.openai_client = OpenAI() if self.provider == 'openai' else None

    def query_language_model(self, prompt, max_tokens=1000):
        if self.provider == 'openai':
            return self.query_openai(prompt, max_tokens)
        elif self.provider == 'huggingface':
            return self.query_huggingface(prompt, max_tokens)
        else:
            raise ValueError("Unsupported language model provider.")

    def query_openai(self, prompt, max_tokens):
        response = self.openai_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.75
        )
        if response.choices:
            last_message = response.choices[-1].message
            return last_message.content if last_message else ''
        return ''

    def query_huggingface(self, prompt, max_tokens):
        # Implementation for Hugging Face API call
        # Placeholder: return a formatted string for now
        return f"Hugging Face response to: {prompt}"