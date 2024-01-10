import json
import os
from typing import Union, List
from openai import OpenAI

from recwizard import BaseModule, monitor
from .tokenizer_chatgpt import ChatgptTokenizer
from .configuration_llm_rec import LLMRecConfig
import logging

logger = logging.getLogger(__name__)


class ChatgptRec(BaseModule):
    """
    The recommender implemented based on OpanAI's GPT models.

    """

    config_class = LLMRecConfig
    tokenizer_class = ChatgptTokenizer

    def __init__(self, config: LLMRecConfig, prompt=None, model_name=None, debug=False, **kwargs):
        """
        Initializes the instance based on the config file.

        Args:
            config (ChatgptRecConfig): The config file.
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
            (ChatgptRec): the instance.
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
            (ChatgptRecTokenizer): the tokenizer.
        """
        # return lambda x: {'context': x}
        return ChatgptTokenizer()

    @monitor
    def response(self, raw_input: str, tokenizer=None, topk=3, max_tokens=None, temperature=0.5, model_name=None,
                 return_dict=False,
                 **kwargs):
        """
        Generate a template to response the processed user's input.

        Args:
            raw_input (dict): A dict that contains the question and its related information.
            tokenizer (BaseTokenizer, optional): A tokenizer to process the question.
            topk (int): The number of answers.
            max_tokens (int): The maximum number of tokens used for ChatGPT API.
            temperature (float): The temperature value used for ChatGPT API.
            model_name (str, optional): The specified GPT model's name.
            return_dict (bool): Whether to return a dict or a list.


        Returns:
            list: The answers.
        """
        if topk == 0:
            return {'output': [], 'links': []} if return_dict else []
        messages = tokenizer(raw_input)['messages']
        messages.append(self.prompt)
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

        try:
            answers = json.loads(res)
            if len(answers) < topk:
                raise ValueError()
            output = [answer['name'] for answer in answers][:topk]
            links = [answer['uri'] for answer in answers][:topk]
        except:
            messages[-1] = self.config.backup_prompt
            res = self.client.chat.completions.create(
                model=self.model_name,
                max_tokens=max_tokens,
                messages=messages,
                temperature=temperature
            ).choices[0].message.content
            output = res.split(',')[:topk]
            if len(output) < topk:
                output = output + [''] * (topk - len(output))
            links = [''] * topk

        if return_dict:
            return {
                "input": messages,
                "output": output,
                "links": {name: link for name, link in zip(output, links)}
            }
        else:
            return output
