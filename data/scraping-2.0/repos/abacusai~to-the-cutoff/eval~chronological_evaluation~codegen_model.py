import openai
import os
import time
from openai.error import RateLimitError
from chronological_evaluation_utils import *

class CodegenModel:

    def config_dict(self):
        return {
            'name': get_name(self)
        }

class OpenAIModel(CodegenModel):

    def __init__(self, args):
        openai.api_key = os.getenv("OPENAI_API_KEY")

    def config_dict(self):
        return {
            'name': get_name(self),
            'model_name': self.MODEL_NAME
        }

class CompleteGPTModel(OpenAIModel):

    def query(self, system_prompt, prompt):

        sampled_response = None
        while sampled_response is None:
            try:
                response = openai.Completion.create(
                    model=self.MODEL_NAME,
                    prompt=f'{system_prompt}\n\n{prompt}',
                    temperature=0.3,
                    max_tokens=60,
                    top_p=1.0,
                    frequency_penalty=0.0,
                    presence_penalty=0.0
                )
                sampled_response = response["choices"][0]["text"]
            except openai.error.RateLimitError:
                time.sleep(5)
            except openai.error.APIError:
                time.sleep(5)
            except openai.error.Timeout:
                time.sleep(5)

        return sampled_response


        # import pdb; pdb.set_trace()

        return sampled_response

class ChatGPTModel(OpenAIModel):

    def query(self, system_prompt, prompt):
        sampled_response = None
        while sampled_response is None:
            try:                
                response = openai.ChatCompletion.create(
                    model=self.MODEL_NAME,
                    messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt},
                        ]
                    )
                sampled_response = response["choices"][0]["message"]["content"]
            except openai.error.RateLimitError:
                time.sleep(5)
                print("error")
            except openai.error.APIError:
                time.sleep(5)
            except openai.error.Timeout:
                time.sleep(5)
            except openai.error.ServiceUnavailableError:
                time.sleep(10)
        return sampled_response


class ChatGPT4Model(ChatGPTModel):

    MODEL_NAME = "gpt-4-0314"


class ChatGPT35TurboModel(ChatGPTModel):

    MODEL_NAME = "gpt-3.5-turbo-0301"


class GPTTextDavinci002Model(CompleteGPTModel):

    MODEL_NAME = "text-davinci-002"
