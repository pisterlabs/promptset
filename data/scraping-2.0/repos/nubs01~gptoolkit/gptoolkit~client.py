"""This module provides a GPT-3 client to query GPT-3 based on user input and use case."""

import keyring
import openai
from gptoolkit.database.database import Database


class GPT3Client:
    """Class to query GPT-3 based on user input and use case.

    Attributes:
        api_key: The GPT-3 API key.
        db: The GPT-3 database.
    """

    def __init__(self, api_key_name, db_file):
        """Creates a new instance of the GPT3Client class.

        Args:
            api_key_name (str): The name of the API key in the keyring.
            db_file (str): The filename of the database to use.
        """
        self.api_key = keyring.get_password('gpt3', api_key_name)
        openai.api_key = self.api_key
        self.db = Database(db_file)

    def query_gpt3(self, user_text, use_case_name):
        """Queries GPT-3 based on user input and use case.

        Args:
            user_text (str): The user input.
            use_case_name (str): The name of the use case.

        Returns:
            str: The response from GPT-3.
        """
        prompts = self.db.query(prompt_text=user_text, use_case_name=use_case_name)

        if not prompts:
            raise ValueError(f"No prompts found for user text '{user_text}' "
                             f"and use case '{use_case_name}'")

        prompt = min(prompts, key=lambda p: p['perplexity'])

        prompt_text = prompt['prompt_text'] + user_text

        response = openai.Completion.create(
            engine=prompt['engine_name'],
            prompt=prompt_text,
            max_tokens=prompt['max_tokens'],
            temperature=prompt['temperature']
        )

        return response.choices[0].text

