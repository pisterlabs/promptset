import openai
import os
import logger
    
from abc import ABC, abstractmethod
import openai

class ModelInterface(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def get_model(self):
        pass

    @abstractmethod
    def get_tokenizer(self):
        pass

    @abstractmethod
    def transform_prompt(self, prompt):
        pass


class GPT4Model(ModelInterface):

    def __init__(self):
        self.openai = openai
        self.openai.api_key = os.getenv("OPENAI_API_KEY")
        self.model_params = {
          'model': "gpt-3.5-turbo",
          'temperature': 0.2,
          'max_tokens': 900,
          'top_p': 1.0,
          'frequency_penalty': 0.0,
          'presence_penalty': 0.0
         }

    def get_model(self):
        return self.model_params

    def get_tokenizer(self):
        tokenizer = openai.GPT4Tokenizer.from_pretrained("openai-gpt4")
        return tokenizer

    def transform_prompt(self, prompt):

        try:
            response = self.openai.ChatCompletion.create(model=self.model_params["model"], messages=[
                {"role": "system", "content": prompt["system"]},
                {"role": "user", "content": prompt["user"]}
                ]
            )

        except Exception as e:
            print(e)
            return {}

        return response.choices[0].message.content

class GPT3Model(ModelInterface):

    def __init__(self):
        self.openai = openai
        self.openai.api_key = os.getenv("OPENAI_API_KEY")
        self.model_params = {
          'model': "text-davinci-003",
          'temperature': 0.2,
          'max_tokens': 900,
          'top_p': 1.0,
          'frequency_penalty': 0.0,
          'presence_penalty': 0.0,
         }
    pass

    def get_model(self):
        return self.model_params

    def get_tokenizer(self):
        tokenizer = openai.GPT4Tokenizer.from_pretrained("openai-gpt4")
        return tokenizer

    def transform_prompt(self, prompt):
        try:
            response = self.openai.Completion.create(**self.model_params, prompt=prompt["user"])
        except Exception as e:
            print(e)
            return {}

        return response.choices[0].text