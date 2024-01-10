from openai import OpenAI
import os


class InferenceService:
    def __init__(self):
        self.__client = OpenAI()
        self.__model = os.getenv('OPENAI_GPT_MODEL', 'text-davinci')
        self.__temperature = float(os.getenv('OPENAI_TEMPERATURE', '1.5'))
        self.__max_tokens = int(os.getenv('OPENAI_MAX_TOKENS', '40'))
        self.__prompt_template = 'Who won GOTY in {}?'

    def __inference(self, prompt: str):
        return self.__client.chat.completions.create(
            model=self.__model,
            messages=[
                {'role': 'user', 'content': prompt}
            ],
            temperature=self.__temperature,
            max_tokens=self.__max_tokens
        )

    def invoke(self, prompt_value: str):
        return self.__inference(
            self.__prompt_template.format(prompt_value)
        )