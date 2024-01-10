import os
from typing import Union, List
from openai import OpenAI

from recwizard import BaseModule, monitor
from .configuration_llm import LLMConfig
from .tokenizer_chatgpt import ChatgptTokenizer
import logging

logger = logging.getLogger(__name__)


class ChatgptGen(BaseModule):
    """
    The generator implemented based on OpanAI's GPT models.

    """

    config_class = LLMConfig
    tokenizer_class = ChatgptTokenizer

    def __init__(self, config: LLMConfig, prompt=None, model_name=None, debug=False,
                 **kwargs):
        """
        Initializes the instance based on the config file.

        Args:
            config (ChatgptGenConfig): The config file.
            prompt (str, optional): A prompt to override the prompt from config file.
            model_name (str, optional): The specified GPT model's name. 
        """

        super().__init__(config, **kwargs)
        self.model_name = config.model_name if model_name is None else model_name
        self.prompt = config.prompt if prompt is None else prompt
        self.debug = debug
        self.client = OpenAI()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, config=None, prompt=None, model_name=None):
        """
        Get an instance of this class.

        Args:
            config:
            pretrained_model_name_or_path:
            prompt (str, optional): The prompt to override the prompt from config file.
            model_name (str, optional): The specified GPT model's name.

        Returns:
             the instance.
        """
        if config is None:
            config = cls.config_class.from_pretrained(pretrained_model_name_or_path)
        return cls(config, prompt=prompt, model_name=model_name)

    def save_pretrained(
            self,
            save_directory: Union[str, os.PathLike],
            push_to_hub: bool = False,
            **kwargs
    ):
        self.config.save_pretrained(save_directory=save_directory, push_to_hub=push_to_hub)

    @classmethod
    def get_tokenizer(cls, **kwargs):
        """
        Get a tokenizer.

        Returns:
            (ChatgptTokenizer): the tokenizer.
        """
        # return lambda x: {'context': x}
        return ChatgptTokenizer()

    @monitor
    def response(self, raw_input, tokenizer, recs: List[str] = None, max_tokens=None, temperature=0.5,
                 model_name=None,
                 return_dict=False,
                 **kwargs):
        """
        Generate a template to response the processed user's input.

        Args:
            raw_input (str): The user's raw input.
            tokenizer (BaseTokenizer, optional): A tokenizer to process the raw input.
            recs (list, optional): The recommended movies.
            max_tokens (int): The maximum number of tokens used for ChatGPT API.
            temperature (float): The temperature value used for ChatGPT API.
            model_name (str, optional): The specified GPT model's name.
            return_dict (bool): Whether to return a dict or a list.

        Returns:
            str: The template to response the processed user's input.
        """

        messages = tokenizer(raw_input)['messages']
        # Add prompt at end
        prompt = self.prompt.copy()
        if recs is not None:
            formatted_movies = ", ".join([f'{i + 1}. "{movie}"' for i, movie in enumerate(recs)])
            prompt['content'] = prompt['content'].format(formatted_movies)
        messages.append(prompt)
        if self.debug:
            logger.info('\ninput:', messages)

        res = self.client.chat.completions.create(
            model=model_name or self.model_name,
            max_tokens=max_tokens,
            messages=messages,
            temperature=temperature
        ).choices[0].message.content

        if self.debug:
            logger.info('\napi result:', res)
        res_out = 'System: {}'.format(res)
        if return_dict:
            return {
                'input': messages,
                'output': res_out,
            }
        return res_out
