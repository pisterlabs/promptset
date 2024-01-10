import datetime
import re
import pandas as pd
from fuzzywuzzy import process

#import logging
from typing import List, Tuple
from .endpoints.openai import OpenAI_API

class TimeExtractor:
    """Extracts time intervals and tokens from a given query using a GPT model."""
    def __init__(self, api_key: str, prompt_filename: str):
        self.api_key = api_key
        self.prompt = self._read_prompt_from_file(prompt_filename)
        self.gpt_model = OpenAI_API(api_key)
    
    def extract_time_interval(self, query: str) -> Tuple[List[datetime.datetime], List[str]]:
        """Returns the datetime interval [start, end] and token list ["token"]."""
        time_reference = datetime.datetime.today()
        time_reference_prompt_str = f'\nRemember the current time is {time_reference}.'
        message = [
            {"role": "user", "content": self.prompt + "\"" + query + "\"" + time_reference_prompt_str}
        ]
        self.gpt_model.user_prompt(message)
        completion = self.gpt_model.get_completion()

        #logging.debug(f'completion: {completion}')

        # Extract the time interval list
        time_interval_list = self._extract_time_interval_list(completion)

        # Extract the symbol list
        symbol_list = self._extract_symbol_list(completion)

        return time_interval_list, symbol_list
    
    def _extract_time_interval_list(self, completion: str) -> List[datetime.datetime]:
        """Extracts the time interval list from the given completion."""

        time_intervals = re.findall(r"\[(.*?)\]", completion)
        time_interval_list = []

        for time_interval_str in time_intervals[:2]:
            datetime_expression = eval(time_interval_str)
            time_interval_list.append(datetime_expression)

        #logging.debug(f'time_interval_list: {time_interval_list}')
        return time_interval_list

    def _extract_symbol_list(self, completion: str) -> List[str]:
        """Extracts the symbol list from the given completion."""

        symbol_match = re.findall(r"\[(.*?)\]", completion)[2]
        symbol_list = [eval(x) for x in symbol_match.split(",")]
        
        #logging.debug(f'symbol_list: {symbol_list}')
        return symbol_list

    def _read_prompt_from_file(self, filename: str) -> str:
        """Reads the prompt from the given file."""

        with open(filename, 'r') as file:
            prompt = file.read()
        return prompt
    
