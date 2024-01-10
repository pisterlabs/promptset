from dotenv import load_dotenv
import sys
import openai
from os import environ


class Environment:
    """
    Holds information about the environment and environment variables.

    Attributes:
        env_file_exists (bool): Whether a .env file exists.
        _API_KEY (str): API Key to be used with OpenAI.
        environment (os.environ): The operating system environment,
            including environment variables.
        args (list[str, ...]): Arguments passed to the script via
            terminal.
        args_length (int): How many arguments were passed to the script
            via terminal.
    """

    def __init__(self):
        """
        Constructs a `Environment` object.
        """
        # Load environment variables
        self.env_file_exists = load_dotenv()
        self.environment = environ
        self._API_KEY = ""
        self.args = sys.argv
        self.args_length = len(self.args)

    def get_api_key(self) -> str:
        """Returns `_API_KEY` attribute."""
        return self._API_KEY

    def _set_api_key(self, api_key: str) -> None:
        """Sets `_API_KEY` attribute."""
        self._API_KEY = api_key

    def set_openai_api_key(self, api_key: str = "") -> None:
        """Sets openai api key and prompts user if one doesn't exist."""
        if api_key:
            # API key was manually passed to method.
            self._set_api_key(api_key)
        elif self.env_file_exists:
            # API key was not manually passed, but a .env file exists
            self._set_api_key(self.environment.get("OPENAI_API_KEY"))
        else:
            # No passed API key and no .env file
            self._set_api_key(input("Enter your api key: "))

        openai.api_key = self.get_api_key()
        self._save_api_key()

    def _save_api_key(self):
        """Saves API key for future use to .env file."""
        with open(".env", 'w', encoding="utf-8") as file:
            print(f'OPENAI_API_KEY="{self._API_KEY}"', file=file)