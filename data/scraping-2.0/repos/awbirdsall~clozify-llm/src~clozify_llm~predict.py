"""predict.py Perform model inference
"""
from abc import abstractmethod

import openai
from openai.api_resources.abstract.engine_api_resource import EngineAPIResource
from openai.openai_object import OpenAIObject
from tenacity import retry, stop_after_attempt, wait_random_exponential

from clozify_llm.constants import (
    DEFAULT_CHAT_MAX_TOKENS,
    DEFAULT_CHAT_MODEL,
    DEFAULT_CHAT_TEMPERATURE,
    END_STR,
    STARTING_MESSAGE,
)
from clozify_llm.utils import format_prompt


class GenericCompleter:
    """
    Base class for getting completion from model.

    Defines standard interface for using some openai EngineAPIResource that can clozify an input when provided with a
    word and (optional) definition, via a call to its `create()` method.

    Sample usage
    ```
    completer = Completer(openai_resource=my_resource)
    cloze_text = completer.get_cloze_text("my_word", "my_definition")
    ```

    Alternatively `get_completion_response()` can be used to return the raw `OpenAIObject` response.

    Parameters
    ----------
    openai_resource : EngineAPIResource
      Resource implementing `create()` method that calls openai API
    model_id : str
      Identifier of model being used
    """

    def __init__(self, openai_resource: EngineAPIResource, model_id: str):
        self.openai_resource = openai_resource
        self.model_id = model_id

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def _create_with_backoff(self, **kwargs):
        """Wrap resource.create call with retry to handle rate limit errors"""
        return self.openai_resource.create(**kwargs)

    @abstractmethod
    def get_completion_response(self, word: str, defn: str, **kwargs) -> OpenAIObject:
        """Get completion response from word and definition.'''"""

    @abstractmethod
    def extract_text_from_response(self, response: OpenAIObject) -> str:
        """Get single text from OpenAI response."""

    def get_cloze_text(self, word: str, defn: str) -> str:
        """Get single cloze completion text from OpenAIObject"""
        completion = self.get_completion_response(word, defn)
        cloze_response = self.extract_text_from_response(completion)
        print(f"response for {word} received, total usage {completion.get('usage').get('total_tokens')}")
        return cloze_response


class Completer(GenericCompleter):
    """Completer for OpenAI "Completion" model"""

    def __init__(self, model_id: str):
        super().__init__(openai_resource=openai.Completion, model_id=model_id)

    def get_completion_response(self, word: str, defn: str, **kwargs) -> OpenAIObject:
        """Get completion response from word and definition

        Includes formatting prompt assumed to be in same way that completion model was fine-tuned.
        """
        prompt = format_prompt(word, defn)
        completion_kwargs = {
            "max_tokens": 200,
            "temperature": 0.2,
        }
        completion_kwargs.update(**kwargs)
        completion = self._get_completion_with_backoff(
            model=self.model_id,
            prompt=prompt,
            stop=END_STR,
            **completion_kwargs,
        )
        return completion

    def extract_text_from_response(self, response: OpenAIObject) -> str:
        return response["choices"][0]["text"].strip()

    def _get_completion_with_backoff(self, model: str, prompt: str, stop: str, **kwargs):
        """Call openai.Completion.create with defined params set"""
        return self._create_with_backoff(model=model, prompt=prompt, stop=stop, **kwargs)


class ChatCompleter(GenericCompleter):
    """Completer for OpenAI "ChatCompleter" model"""

    def __init__(self, model_id: str = DEFAULT_CHAT_MODEL):
        super().__init__(openai_resource=openai.ChatCompletion, model_id=model_id)

    def get_completion_response(self, word: str, defn: str, **kwargs) -> OpenAIObject:
        """Get completion response from word

        Note: definition ignored at present

        Parameters
        ----------
        word : str
          Word to be clozified.
        defn : str
          Definition of word to be clozified (note: currently ignored).
        **kwargs
          Additional kwargs passed to self._make_chat_params() (temperature, max_tokens).
        """
        chat_params = self._make_chat_params(input_word=word, **kwargs)
        response = self._create_with_backoff(**chat_params)
        return response

    def extract_text_from_response(self, response: OpenAIObject) -> str:
        return response["choices"][0]["message"]["content"].strip()

    def _make_chat_params(
        self,
        input_word: str,
        temperature: float = DEFAULT_CHAT_TEMPERATURE,
        max_tokens: int = DEFAULT_CHAT_MAX_TOKENS,
    ) -> dict:
        """Assemble parameters for openai.ChatCompletion request"""
        prompt_message = {"role": "user", "content": f"Input: {input_word}"}
        messages = STARTING_MESSAGE + [prompt_message]
        return {"model": self.model_id, "temperature": temperature, "max_tokens": max_tokens, "messages": messages}
