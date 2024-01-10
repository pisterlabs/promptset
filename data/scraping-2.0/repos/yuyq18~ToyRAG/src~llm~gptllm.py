from .base import BaseLLM
from argparse import ArgumentParser
from loguru import logger
import openai
from langchain import OpenAI
import json
from ..utils import *
import tiktoken

class GPTLLM(BaseLLM):
    @staticmethod
    def parse_llm_args(parser: ArgumentParser) -> ArgumentParser:
        parser.add_argument('--api_config', type=str, default='config/api-config.json', help='Configuration file for llm')
        
        return parser
    
    def __init__(self):
        parser = ArgumentParser()
        parser = self.parse_llm_args(parser)
        args, extras = parser.parse_known_args()

        with open(args.api_config, 'r') as f:
            self.api_config = json.load(f)
            openai.api_base = self.api_config['api_base']
        logger.success(args)
        self.model = OpenAI(
            model_name=self.api_config['model'],
            temperature=self.api_config['temperature'],
            max_tokens=self.api_config['max_tokens'],
            model_kwargs={"stop": "\n"},
            openai_api_key=self.api_config['api_key'],
        )
        self.enc = tiktoken.encoding_for_model(self.api_config['model'])
    
    def step(self, question, retrieval_res):
        llm_input = question+"\nHere is some reference information:\n" + retrieval_res
        if len(self.enc.encode(llm_input)) >= 3096:
            llm_input = llm_input[:14000]
        logger.info(llm_input)
        logger.debug(f'LLM input length: {len(self.enc.encode(llm_input))}')

        llm_response = self.model(question) # without reference information
        llm_response_ref = self.model(llm_input) # with reference information
        logger.debug(f'LLM output length: {len(self.enc.encode(llm_response))}')
        return format_text(llm_response), format_text(llm_response_ref)
