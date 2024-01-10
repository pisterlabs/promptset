import sys
import openai
import json
from src.gpt import GPT


class Client:
    """A class representing the OpenAI API Client"""
    def __init__(self, config):
        self.client = None
        self.api_key = ''
        self.config = config

    def initialize(self):
        """Checks config.json for a stored API key, or prompts the user to input a new key"""
        config_data = self._json_load(self.config)
        api_key = config_data['api_key']
        if api_key:
            good_key = self.test_key(api_key)
            if good_key:
                self.api_key = api_key
                self.client = openai.OpenAI(api_key=self.api_key)
            else:
                self.set_key()
        else:
            self.set_key()

    @staticmethod
    def test_key(api_key):
        """Send a test message to the GPT API to check if an API key is valid"""
        client = openai.OpenAI(api_key=api_key)
        try:
            try:
                response = client.chat.completions.create(
                    model='gpt-3.5-turbo',
                    max_tokens=5,
                    messages=[{'role': 'user', 'content': 'This is a test .'}])
            except openai.AuthenticationError:
                print('[*] error, invalid API key')
                return False
            else:
                print('[*] API key verified')
                return True
        except openai.APIConnectionError:
            print('[*] network connection error\n[*] exiting')
            sys.exit()

    def set_key(self):
        """Set a new API key and test if it is valid"""
        while True:
            self.api_key = input('[*] insert OpenAI API key:\n>')
            valid_key = self.test_key(self.api_key)
            if valid_key:
                config_data = self._json_load(self.config)
                config_data['api_key'] = self.api_key
                self._json_dump(config_data, self.config)
                self.client = openai.OpenAI(api_key=self.api_key)
                return

    @staticmethod
    def _json_load(json_file):
        """Loads JSON object from a file"""
        with open(json_file, 'r') as f:
            data = json.load(f)
        return data

    @staticmethod
    def _json_dump(json_dict, json_file):
        """Dumps a JSON object to a file"""
        with open(json_file, 'w') as f:
            json.dump(json_dict, f, indent=6)

