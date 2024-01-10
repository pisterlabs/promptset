import os
import logging
from openai import OpenAI
from config import Config

class MissingAPIKeyError(Exception):
    def __init__(self):
        message = "OpenAI API key not found in environment variables. Please set the OPENAI_API_KEY environment variable."
        super().__init__(message)

class GPTSuggestions:
    def __init__(self, config):
        self.logger = logging.getLogger(__name__)
        logging_level = getattr(logging, config.debug_level, logging.WARNING)
        self.logger.setLevel(logging_level)
        self.config = config
        self.api_key = os.getenv('OPENAI_API_KEY')
        if self.api_key is None:
            raise MissingAPIKeyError()

        # Instantiate the OpenAI client
        self.client = OpenAI(api_key=self.api_key)

        # Get the language model type from config
        self.model_type = config.get('OpenAI_integration', 'language_model')
        self.model = "gpt-4" if self.model_type == "GPT-4" else "gpt-3.5-turbo"

    def get_suggestion(self, word, context):
        try:
            # Setting up the prompt for GPT
            system_message = {
                "role": "system",
                "content": "You are a seasoned scholar in the humanities specialized in early modern languages. Your role is to help decipher abbreviations from Renaissance and early modern printed books. Provide a concise, comma-separated list of possible word suggestions for the following problematic word and context."
            }
            user_message = {
                "role": "user",
                "content": f"Problematic Word: {word}\nContext: {context}"
            }

            # Making the API call
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[system_message, user_message],
                max_tokens=50
            )

            # Log the full API response
            self.logger.info(f"API Response: {response}")

            # Checking and extracting the response
            if response.choices:
                # Directly accessing attributes of the Choice object
                message_content = response.choices[0].message.content if response.choices[0].message else ""
                # Assuming the first line contains the comma-separated word suggestions
                return message_content.split('\n')[0]
            else:
                return "No suggestions available"
        except Exception as e:
            self.logger.exception("Error in generating GPT suggestion")
            return f"Error in generating suggestion: {e}"


    def print_suggestion(self, context, suggestion):
        print("\n\n")
        print(f"[bold]GPT Suggestion:[/bold] {suggestion}")
        print("\n\n")

    def get_and_print_suggestion(self, context):
        suggestion = self.get_suggestion(context)
        self.print_suggestion(context, suggestion)
        return suggestion
