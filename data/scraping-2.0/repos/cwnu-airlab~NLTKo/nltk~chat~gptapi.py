import os
import openai
from typing import List, Dict, Tuple, Union
import time
import datetime
import json
import pickle


class GPT_sentence_maker:
    def __init__(self, key: str = "-1", max_tokens: int = 800, temperature: float = 0.2) -> None:
        self.key = key
        self.max_tokens = max_tokens
        self.temperature = temperature
        if self.key == "-1":
            raise ValueError("Please enter your key")

    def _make_sentence(self, prompt, message) -> dict:
        """
        prompt: str
        message: str
        """
        openai.api_key = self.key

        send_dict = [
            {
            "role": "system",
            "content": prompt
            },


            {
            "role": "user",
            "content": message
            },
        ]
        
        response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=send_dict,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
    
        # result = response["choices"][0]["message"]['content']

        result = response

        return result


class GPTSentence(GPT_sentence_maker):
    def __init__(self, key: str = "-1", max_tokens: int = 800, temperature: float = 0.2) -> None:
        super().__init__(key=key, max_tokens=max_tokens, temperature=temperature)

        self.prompt = """
        this is default prompt

         """
        
    def make_sentence(self, prompt: str, message: str, return_type: str = "json"):
        
        return_type_list = ['pickle', 'json']

        self.prompt = prompt


        result = self._make_sentence(self.prompt, message)

        if return_type not in return_type_list:
            raise ValueError("Please enter correct return_type")
        elif return_type == "pickle":
            result = pickle.dumps(result)
        elif return_type == "json":
            result = json.dumps(result)
            result = json.loads(result)
        
        return result

    def _remove_numbering(self, returned_dict: dict) -> dict:
        content_list = returned_dict['content']
        result_list = []
        for str1 in content_list:
            point_index = str1.find('.', 0, 4)
            if point_index != -1:
                result_list.append(str1[point_index+1:])
            else:
                result_list.append(str1)
        
        returned_dict['content'] = result_list
        return returned_dict

    def _remove_trash(self, returned_dict: dict) -> dict:
        content_list = returned_dict['content']
        result_list = []
        trash_list = ['-', '*']
        for str1 in content_list:
            for trash in trash_list:
                point_index = str1.find(trash, 0, 4)
                if point_index != -1:
                    str1 = str1[point_index+1:]

            result_list.append(str1)
        
        returned_dict['content'] = result_list
        return returned_dict

    def _only_sentence(self, returned_dict: dict) -> dict:
        content_list = returned_dict['content']
        result_list = []
        for str1 in content_list:
            if str1[-1] == '.':
                result_list.append(str1)
            
        
        returned_dict['content'] = result_list
        return returned_dict

    def _remove_blank(self, returned_dict: dict) -> dict:
        content_list = returned_dict['content']
        result_list = []
        for str1 in content_list:
            result_list.append(str1.strip())
            
        
        returned_dict['content'] = result_list
        return returned_dict

    def check_error_sentences(self, input: str) -> bool:
        if input[0] == '{':
            try:
                json.loads(input)
                return True
            except:
                return False
        else:
            return False

    def make_sent_in_format(self, input: Union[dict, str]) -> dict:
        if type(input) == str:
            json.loads(input)
        input = self._remove_numbering(input)
        input = self._remove_trash(input)
        input = self._only_sentence(input)
        input = self._remove_blank(input)
        return input

class TextReader_:
    def __init__(self, path: str):
        self.path = path

    def read(self) -> List[str]:
        with open(self.path, 'r') as f:
            return f.readlines()




