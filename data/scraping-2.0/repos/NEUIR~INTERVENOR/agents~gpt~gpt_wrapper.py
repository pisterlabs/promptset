import backoff
import openai
import os
import numpy as np
# os.environ["http_proxy"]="127.0.0.1:7890"
# os.environ["https_proxy"]="127.0.0.1:7890"
# openai.api_key = "Put your API key here"

# api_list = [
# "Put your API key1 here",
# "Put your API key2 here"
# ...
# ]
api_nums = len(api_list)
class GPTAgent():
    def __init__(self, model):
        """
        :param model: The model to use for completion.
        """
        self.model = model


    @backoff.on_exception(
        backoff.fibo,
        # https://platform.openai.com/docs/guides/error-codes/python-library-error-types
        (
            openai.error.APIError,
            openai.error.Timeout,
            openai.error.RateLimitError,
            openai.error.ServiceUnavailableError,
            openai.error.APIConnectionError,
            KeyError,
        ),
    )
    def __call__(self, user_prompt, max_tokens, temperature, stop_words):
        """
        :param user_prompt: The user requirments.
        :param max_tokens: The maximum number of tokens to generate.
        :param temperature: The sampling temperature.
        :param stop: The stop sequence or a list of stop sequences.
        :return: Return the response and the usage of the model.
        """
        api_key = api_list[np.random.randint(0, api_nums)]
        openai.api_key = api_key
        response = openai.Completion.create(engine=self.model, prompt=user_prompt, max_tokens=max_tokens, temperature=temperature,
                                 stop=stop_words)
        return response.choices[0].text, response["usage"]


