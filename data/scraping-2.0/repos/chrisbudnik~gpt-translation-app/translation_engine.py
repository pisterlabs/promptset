import os
from typing import List, Literal
import openai 
from openai.openai_object import OpenAIObject
from languages import SupportedLanguages
from google.cloud import translate_v2 as translate
from google.cloud import aiplatform


class TranslationEngine:
    def __init__(
            self, 
            countries: List[str], 
            engine = Literal["OpenAI", "Google"]
        ) -> None:

        self.countries = countries
        self.engine = engine

        if self.engine == "OpenAi":
            self.api_key = os.getenv("OPENAI_API_KEY")
            openai.api_key = self.api_key

        if self.engine == "Google":
            self.translate_client = translate.Client()

    @staticmethod
    def _process_openai_response(response: OpenAIObject) -> str:
        """Processes openai chat response and saves message output as string."""

        response_dict: dict = response.to_dict_recursive()
        response_choice: dict[str: dict] = response_dict.get("choices", {})[0]
        return response_choice.get("message", {}).get("content", None)

    def get_openai_response(self, context: str, prompt: str, **kwargs) -> str:
        """Query openai api for chat response based on provided context, prompt and extra kwargs."""

        response: OpenAIObject = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": context},
                {"role": "user", "content": prompt},
            ],
            **kwargs
            )
        return self._process_openai_response(response)
    
    def translate_with_openai(self, text: str, target_language: str) -> str:
        """Translate text with openai API."""

        # Set up the context for translation - change to template later
        context = f"Translate the following English text to {target_language}:"
        
        # Call the OpenAI API
        translated_text = self.get_openai_response(context, text)
        return translated_text
    
    def translate_with_google(self, text: str, target_language):
        """Translate text with google translate API."""

        # google translate
        result = self.translate_client.translate(text, target_language)
        return result['translatedText']
    

    def generate_translations(self, text: str) -> dict:
        """Generate translations for all supported languages."""

        translations = {}
        for country in self.countries:
            language = SupportedLanguages.REFERENCE.get(country)
            header_key = f"{country} ({language})"

            if self.engine == "OpenAI":
                translations[header_key] = self.translate_with_openai(text, language)
            
            elif self.engine == "Google":
                translations[header_key] = self.translate_with_google(text, country)
            
            else:
                raise NotImplementedError(f"Translation engine not implemented yet.")
        
        return translations
    

