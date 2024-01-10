import os
import openai
from typing import Any, List, Dict, Tuple


class GPTClient:
    def __init__(
        self,
        model: str,
        api_key: str,
        max_tokens: int,
        elevenlabs_key: str,
        FLAG: Dict,
        use_vosk: bool,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.elevenlabs_key = elevenlabs_key
        openai.api_key = self.api_key
        self.flags = FLAG
        self.use_vosk = use_vosk

    def send_and_recv(
        self, msg: List[Dict[str, Any]], temp: float, out_num: int
    ) -> Tuple[List[str], Exception]:
        """
        Sends a list of messages to the GPT-3 model and returns a list of generated responses and any exceptions raised.

        Args:
        - messages: A list of dictionaries representing chat prompts.
        - temperature: A float controlling the randomness of the generated responses. Higher values lead to more random responses.
        - output_num: The number of responses to generate.

        Returns:
        - A tuple containing two elements:
            - A list of generated responses.
            - An exception, if any was raised during the API call.
        """

        self.set_proxy()
        respond = openai.ChatCompletion.create(
            model=self.model,
            messages=msg,
            max_tokens=self.max_tokens,
            temperature=temp,
            # top_p=1,
            # frequency_penalty=0,
            # presence_penalty=0,
            # stop=["\n", " Human:", " AI:"],
        )
        self.unset_proxy()
        result = [respond.choices[0].message["content"]]
        err = None
        return result, err

    def set_proxy(self) -> None:
        """
        Sets the HTTP and HTTPS proxy environment variables to use a local proxy server.
        """
        os.environ["https_proxy"] = "http://127.0.0.1:7890"
        os.environ["http_proxy"] = "http://127.0.0.1:7890"

    def unset_proxy(self) -> None:
        """
        Unsets the HTTP and HTTPS proxy environment variables.
        """
        os.environ["https_proxy"] = ""
        os.environ["http_proxy"] = ""
