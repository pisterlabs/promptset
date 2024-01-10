from typing import List, Dict, Tuple
import openai
from openai.openai_object import OpenAIObject


class RequestChatGPT:
    def __init__(self):
        pass

    @staticmethod
    async def chat_completions(
        model: str,
        messages: List[Dict[str, str]],
        # timeout: int = 15,
        # request_timeout: int = 15,
        **kwargs
    ) -> Tuple[str, int]:
        """Request message to ChatGPT model for chat completions.
        API reference: https://platform.openai.com/docs/api-reference/chat

        Parameters
        ----------
        model : str
            model name
        messages : List[Dict[str, str]]
            messages to be sent to ChatGPT
        opts : Dict
            Important options.
            - max_tokens (int):
                The maximum number of tokens allowed for the generated answer.
                By default, the number of tokens the model can return will be (4096 - prompt tokens).
            - n (int):
                How many chat completion choices to generate for each input message.
            - temperature (float):
                What sampling temperature to use, between 0 and 2.
                Higher values like 0.8 will make the output more random,
                while lower values like 0.2 will make it more focused and deterministic.

        Examples
        --------
        >>> RequestChatGPT.chat_completions(
        >>>     model="gpt-3.5-turbo",
        >>>     messages=[
        >>>           {"role": "system", "content": "You are a helpful assistant."},
        >>>           {"role": "user", "content": "Who won the world series in 2020?"},
        >>>           {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
        >>>           {"role": "user", "content": "Where was it played?"}
        >>>       ]
        >>> )
        {
          "choices": [
            {
              "finish_reason": null,
              "index": 0,
              "message": {
                "content": "The 2020 World Series was ...",
                "role": "assistant"
              }
            }
          ],
          "created": 1677913033,
          "id": "chatcmpl-6qG7N1qgU9WtmGpJZZZB9RRwwU7of",
          "model": "gpt-3.5-turbo-0301",
          "object": "chat.completion",
          "usage": {
            "completion_tokens": 40,
            "prompt_tokens": 56,
            "total_tokens": 96
          }
        }

        Raises
        ------
        TypeError
            If the type of response is not OpenAIObject.

        Returns
        -------
        Dict
            The Response for sended message.

        """
        res = openai.ChatCompletion.create(
            model = model,
            messages = messages,
            # timeout = timeout,
            # request_timeout = request_timeout,
            **kwargs,
        )
        if isinstance(res, OpenAIObject):
            return res["choices"][0]["message"]["content"], res["usage"]["total_tokens"]
        else:
            raise TypeError("The type of response was expected OpenAIObject, but got {type(res)}")

    @staticmethod
    def generate_code(instruction: str, model: str = "gpt-3.5-turbo") -> str:
        """Generate code according to instruction via ChatGPT.

        Parameters
        ----------
        instruction : str
            message to be attached to the request

        Returns
        -------
        str
            generated code by ChatGPT

        """
        messages = [
            {"role": "system", "content": (
                "You should generate the code according to the instructions given."
                "The reply should be the code part only."
            )},
            {"role": "user", "content": instruction}
        ]
        res = RequestChatGPT.chat_completions(model=model, messages=messages)
        return str(res["choices"][0]["message"]["content"])
