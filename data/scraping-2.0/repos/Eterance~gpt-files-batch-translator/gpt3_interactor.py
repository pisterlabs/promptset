from enum import Enum
import logging
import os
import random
import re
import traceback
from typing import Any
import openai
from openai.error import RateLimitError, APIConnectionError, ServiceUnavailableError, APIError, InvalidRequestError
from transformers import GPT2TokenizerFast
import time


class CompletionTypeEnum(Enum):
    ChatCompletion = "ChatCompletion"
    Completion = "Completion"
    
# usage limit:
# https://help.openai.com/en/articles/5955598-is-api-usage-subject-to-any-rate-limits
class Gpt3Interactor():
    """
    Class that interact with openai gpt3 model.
    """
    # TODO: 修改所有引用，因为 model_name 参数已被删除
    def __init__(self, api_keys:list[str], logger:logging.Logger=None, shuffle_keys:bool=True, **useless_args):
        self.keys = api_keys
        if shuffle_keys:
            random.shuffle(self.keys)
        self.current_key_id = 0
        # prevent logger lots of logs
        self._openai_logger = logging.getLogger("openai")
        self._openai_logger.setLevel(logging.WARNING)
        
    def calculate_token_counts(self, string:str):
        """
        Calculate the token counts of the string using GPT2TokenizerFast.
        Note: this is for reference only, and maybe not accurate for codex (because codex uses a different tokenizer).
        see: https://beta.openai.com/tokenizer
        """
        tokenizer:GPT2TokenizerFast = GPT2TokenizerFast.from_pretrained("gpt2")
        return len(tokenizer(string)['input_ids'])
    
    # TODO: 修改之前所有引用到这个函数的地方，因为参数已经改变
    def generate(
            self,
            engine: str,
            prompt: str|list[dict],
            max_tokens:int,
            n: int = 1,
            temperature: float = 0,
            top_p: float = 1,
            stop: list[str] = ['--', '\n\n', ';', '#'],
            error_wait_time: int = 5,
            n_batch_size: int = 4,
            completion_type:CompletionTypeEnum = CompletionTypeEnum.Completion
    ):
        """
        args:
            n_batch_size(int): if n too large, rate may over the limit (40000 tokens per min) and cause RateLimitError.
                               So split n into n_batch_size and call api multiple times, then merge the results.
                               If n_batch_size is too small, it will slow down the process,
                               but if n_batch_size is too large, it will cause RateLimitError.
                               Set n_batch_size < 1 to disable this feature.
                               For table_fact, recommend n_batch_size=10~12, and 4~6 for wikitq.
        """
        if n > n_batch_size and n_batch_size > 0:
            # split into a list, contains multiple n=n_batch_size and last one n=n%n_batch_size
            n_list:list[int] = [n_batch_size] * (n // n_batch_size)
            if n % n_batch_size != 0:
                n_list.append(n % n_batch_size)
            result_list = []
            for new_n in n_list:
                result = self.generate(
                    engine=engine,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    n=new_n,
                    stop=stop,
                    error_wait_time=error_wait_time,
                    n_batch_size=n_batch_size
                )
                result_list.append(result)
            return self._merge_multiple_response_dicts(result_list)
        else:
            start_time = time.time()
            result = None
            while result is None:
                try:
                    key = self.keys[self.current_key_id]
                    if len(self.keys) <= 0:
                        print(f"!!!!!!!!!!!!!!!!!No Key available!!!!!!!!!!!!!!")
                        raise Exception("No openai api key available.")
                    self.current_key_id = (self.current_key_id + 1) % len(self.keys)
                    print(f"Using openai api key: {key}")
                    #print(f"Using openai api key: {key}")
                    if completion_type == CompletionTypeEnum.ChatCompletion:
                        assert isinstance(prompt, list), "prompt/messages must be a list of dict when using ChatCompletion."
                        if stop is not None:
                            result = openai.ChatCompletion.create(
                                model=engine,
                                messages=prompt,
                                api_key=key,
                                max_tokens=max_tokens,
                                temperature=temperature,
                                top_p=top_p,
                                n=n,
                                stop=stop
                            )
                        else:
                            result = openai.ChatCompletion.create(
                                model=engine,
                                messages=prompt,
                                api_key=key,
                                max_tokens=max_tokens,
                                temperature=temperature,
                                top_p=top_p,
                                n=n
                            )
                    else:
                        if stop is not None:
                            result = openai.Completion.create(
                                engine=engine,
                                prompt=prompt,
                                api_key=key,
                                max_tokens=max_tokens,
                                temperature=temperature,
                                top_p=top_p,
                                n=n,
                                stop=stop,
                                logprobs=1
                            )
                        else: 
                            result = openai.Completion.create(
                                engine=engine,
                                prompt=prompt,
                                api_key=key,
                                max_tokens=max_tokens,
                                temperature=temperature,
                                top_p=top_p,
                                n=n,
                                logprobs=1
                            )
                    print(f'Openai api inference time: {time.time() - start_time}')
                    #print('Openai api inference time:', time.time() - start_time)
                    return result
                except RateLimitError as rte:
                    if "You exceeded your current quota" in str(rte):
                        print(f"key {key} exceeded current quota. Will remove from keys (remain {len(self.keys)-1}) and retry.")
                        self.keys.remove(key)
                        self.current_key_id -= 1
                        time.sleep(error_wait_time)
                    elif n_batch_size >= 2:
                        # Perhaps n_batch_size still too large, reduce to half
                        print(f"{rte}, n_batch_size: {n_batch_size} -> {n_batch_size//2} and retry.")
                        #print(f"{rte}, n_batch_size: {n_batch_size} -> {n_batch_size//2} and retry.")
                        result = self.generate(
                            engine=engine,
                            prompt=prompt,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            top_p=top_p,
                            n=n_batch_size,
                            stop=stop,
                            error_wait_time=error_wait_time,
                            n_batch_size=n_batch_size//2
                        )
                        return result
                    else:
                        # It means request is too frequent, so just retry
                        print(f'{rte} n_batch_size: {n_batch_size}. Retry.')
                        #print(f'{rte} n_batch_size: {n_batch_size}. Retry.')
                        time.sleep(error_wait_time)
                except InvalidRequestError as ire: 
                    if ire.code == 'context_length_exceeded': 
                        # extract the context length from the error message
                        max_length = self._extract_numbers(str(ire), r"This model's maximum context length is (\d+) tokens")
                        request_length = self._extract_numbers(str(ire), r"you requested (\d+) tokens")
                        old_max_tokens = max_tokens
                        max_tokens = max_length - request_length
                        print(f'{type(ire)}: context length exceeded, max_tokens {old_max_tokens}->{max_tokens} and retry.')
                    else:
                        print(f'{type(ire)}: {ire}. Retry.')
                        raise ire
                except APIError as apie:
                    if apie.http_status is not None and apie.http_status == 500:\
                        print(f'{type(apie)}: {apie}, Retry.')
                    #print(e, 'Retry.')
                    time.sleep(error_wait_time)
                except Exception as e:
                    traceback.print_exc()
                    print(f'\033[1;31m (From Gpt3Interactor) 红色\033[0m {type(e)}: {e}, Retry.')
                    #print(e, 'Retry.')
                    time.sleep(error_wait_time)
                    #time.sleep(5)
    
    def _extract_numbers(self, string, pattern):
        match = re.search(pattern, string)
        if match:
            return int(match.group(1))
        else:
            return None
    
    def _merge_multiple_response_dicts(self, response_dicts: list[dict])->dict:       
        response_dict = response_dicts[0]
        for response_dict_ in response_dicts[1:]:
            response_dict['choices'].extend(response_dict_['choices'])
        return response_dict