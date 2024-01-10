import os
from dotenv import load_dotenv
import openai

# Load environment variables from .env file
load_dotenv()

class AICore:
    def __init__(self):
        self.api_key = os.getenv('OPENAI_API_KEY')
        self.default_language = os.getenv('DEFAULT_LANGUAGE', 'en')
        self.max_tokens = int(os.getenv('MAX_TOKENS', 256))
        self.openai_client = openai.Completion()

        # Ensure the OpenAI API key is set
        if not self.api_key:
            raise ValueError("The OpenAI API key must be set in the environment variables.")

        # Configure OpenAI with the API key
        openai.api_key = self.api_key

    def generate_text(self, prompt, language=None, max_tokens=None):
        """
        Generate text using the OpenAI API based on the given prompt.

        :param prompt: The input text prompt to generate content from.
        :param language: The language for content generation (if multilingual support is enabled).
        :param max_tokens: The maximum number of tokens to generate.
        :return: The generated text as a string.
        """
        language = language if language else self.default_language
        max_tokens = max_tokens if max_tokens else self.max_tokens

        # Ensure the language is supported if multilingual support is enabled
        supported_languages = os.getenv('SUPPORTED_LANGUAGES')
        if supported_languages and language not in supported_languages.split(','):
            raise ValueError(f"Language '{language}' is not supported.")

        try:
            response = self.openai_client.create(
                engine="davinci",
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=0.7,
                top_p=1,
                n=1,
                stop=None,
                language=language
            )
            return response.choices[0].text.strip()
        except openai.error.OpenAIError as e:
            raise Exception(f"An error occurred while generating text: {str(e)}")

# Example usage:
# ai = AICore()
# generated_text = ai.generate_text("Write a creative description for a smartwatch that appeals to tech enthusiasts.")
# print(generated_text)
