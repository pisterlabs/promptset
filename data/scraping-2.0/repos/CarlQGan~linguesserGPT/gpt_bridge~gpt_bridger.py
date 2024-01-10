import os
import openai


"""
--------------
Helper methods
--------------
"""

def __read_prompt(prompt_path):
    with open(prompt_path, 'r') as file:
        prompt = file.read()
    return prompt

prompt_path = "./gpt_bridge/prompt.txt"  # file path to the prompt formatter
PROMPT_FORMAT = __read_prompt(prompt_path)


"""
A bridger class to communicate with OpenAI's GPT API.
"""
class GPTBridger:
    def __init__(self, api_key : str = None) -> None:
        self.api_key = api_key
        
        if self.api_key is None:
            self.api_key = os.getenv("OPENAI_API_KEY")
        if self.api_key is None:
            print("No available OpenAI API Key found in the environment. Please input your OpenAI API Key below, or set your own OPENAI_API_KEY in environment variables.")
            self.api_key = input("Please input your OpenAI API Key: ")
        if self.api_key is None:
            raise ValueError("No available OpenAI API Key received.")
        
        openai.api_key = self.api_key
        if not self._has_valid_api_key():
            raise ValueError("Invalid OpenAI API Key.")
        
        self.current_language = None

    def get_prompt_response(self, prompt : str) -> str:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
        )
        return response['choices'][0]['message']['content'].strip()
        
    def get_next_language_example(self, language : str) -> str:
        if language is None:
            raise ValueError("No language selected.")
        self.set_current_language(language)
        return self.get_prompt_response(PROMPT_FORMAT.format(language=language)).replace(language, "").strip()
    
    def get_same_language_example(self) -> str:
        if self.get_current_language() is None:
            raise ValueError("No language selected.")
        return self.get_prompt_response(PROMPT_FORMAT.format(language=self.get_current_language())).replace(self.get_current_language(), "").strip()
    
    def get_current_language(self) -> str:
        return self.current_language

    def set_current_language(self, language : str) -> None:
        self.current_language = language
    
    def _has_valid_api_key(self) -> bool:
        try:
            openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "test"}],
                max_tokens=5,
            )
            return True
        except openai.error.AuthenticationError:
            return False
