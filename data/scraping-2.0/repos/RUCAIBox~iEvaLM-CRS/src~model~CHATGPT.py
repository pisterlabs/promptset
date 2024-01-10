import typing
import torch
import json
import os

import nltk
import openai
import tiktoken
import numpy as np

from loguru import logger
from sklearn.metrics.pairwise import cosine_similarity
from accelerate.utils import set_seed
from tenacity import _utils, Retrying, retry_if_not_exception_type
from tenacity.stop import stop_base
from tenacity.wait import wait_base
from thefuzz import fuzz
from tqdm import tqdm

def my_before_sleep(retry_state):
    logger.debug(f'Retrying: attempt {retry_state.attempt_number} ended with: {retry_state.outcome}, spend {retry_state.seconds_since_start} in total')


class my_wait_exponential(wait_base):
    def __init__(
        self,
        multiplier: typing.Union[int, float] = 1,
        max: _utils.time_unit_type = _utils.MAX_WAIT,  # noqa
        exp_base: typing.Union[int, float] = 2,
        min: _utils.time_unit_type = 0,  # noqa
    ) -> None:
        self.multiplier = multiplier
        self.min = _utils.to_seconds(min)
        self.max = _utils.to_seconds(max)
        self.exp_base = exp_base

    def __call__(self, retry_state: "RetryCallState") -> float:
        if retry_state.outcome == openai.error.Timeout:
            return 0

        try:
            exp = self.exp_base ** (retry_state.attempt_number - 1)
            result = self.multiplier * exp
        except OverflowError:
            return self.max
        return max(max(0, self.min), min(result, self.max))


class my_stop_after_attempt(stop_base):
    """Stop when the previous attempt >= max_attempt."""

    def __init__(self, max_attempt_number: int) -> None:
        self.max_attempt_number = max_attempt_number

    def __call__(self, retry_state: "RetryCallState") -> bool:
        if retry_state.outcome == openai.error.Timeout:
            retry_state.attempt_number -= 1
        return retry_state.attempt_number >= self.max_attempt_number

def annotate(conv_str):
    request_timeout = 6
    for attempt in Retrying(
        reraise=True, retry=retry_if_not_exception_type((openai.error.InvalidRequestError, openai.error.AuthenticationError)),
        wait=my_wait_exponential(min=1, max=60), stop=(my_stop_after_attempt(8)), before_sleep=my_before_sleep
    ):
        with attempt:
            response = openai.Embedding.create(
                model='text-embedding-ada-002', input=conv_str, request_timeout=request_timeout
            )
        request_timeout = min(30, request_timeout * 2)

    return response

def annotate_chat(messages, logit_bias=None):
    if logit_bias is None:
        logit_bias = {}

    request_timeout = 20
    for attempt in Retrying(
        reraise=True, retry=retry_if_not_exception_type((openai.error.InvalidRequestError, openai.error.AuthenticationError)),
        wait=my_wait_exponential(min=1, max=60), stop=(my_stop_after_attempt(8)), before_sleep=my_before_sleep
    ):
        with attempt:
            response = openai.ChatCompletion.create(
                model='gpt-3.5-turbo', messages=messages, temperature=0, logit_bias=logit_bias,
                request_timeout=request_timeout,
            )['choices'][0]['message']['content']
        request_timeout = min(300, request_timeout * 2)

    return response

class CHATGPT():
    
    def __init__(self, seed, debug, kg_dataset) -> None:
        self.seed = seed
        self.debug = debug
        if self.seed is not None:
            set_seed(self.seed)
        
        self.kg_dataset = kg_dataset
        
        self.kg_dataset_path = f"../data/{self.kg_dataset}"
        with open(f"{self.kg_dataset_path}/entity2id.json", 'r', encoding="utf-8") as f:
            self.entity2id = json.load(f)
        with open(f"{self.kg_dataset_path}/id2info.json", 'r', encoding="utf-8") as f:
            self.id2info = json.load(f)
            
        self.id2entityid = {}
        for id, info in self.id2info.items():
            if info['name'] in self.entity2id:
                self.id2entityid[id] = self.entity2id[info['name']]
        
        self.item_embedding_path = f"../save/embed/item/{self.kg_dataset}"
        
        item_emb_list = []
        id2item_id = []
        for i, file in tqdm(enumerate(os.listdir(self.item_embedding_path))):
            item_id = os.path.splitext(file)[0]
            if item_id in self.id2entityid:
                id2item_id.append(item_id)

                with open(f'{self.item_embedding_path}/{file}', encoding='utf-8') as f:
                    embed = json.load(f)
                    item_emb_list.append(embed)

        self.id2item_id_arr = np.asarray(id2item_id)
        self.item_emb_arr = np.asarray(item_emb_list)
            
        self.chat_recommender_instruction = '''You are a recommender chatting with the user to provide recommendation. You must follow the instructions below during chat.
If you do not have enough information about user preference, you should ask the user for his preference.
If you have enough information about user preference, you can give recommendation. The recommendation list must contain 10 items that are consistent with user preference. The recommendation list can contain items that the dialog mentioned before. The format of the recommendation list is: no. title. Don't mention anything other than the title of items in your recommendation list.'''
    
    def get_rec(self, conv_dict):
        
        rec_labels = [self.entity2id[rec] for rec in conv_dict['rec'] if rec in self.entity2id]
        
        context = conv_dict['context']
        context_list = [] # for model
        
        for i, text in enumerate(context):
            if len(text) == 0:
                continue
            if i % 2 == 0:
                role_str = 'user'
            else:
                role_str = 'assistant'
            context_list.append({
                'role': role_str,
                'content': text
            })
        
        conv_str = ""
        
        for context in context_list[-2:]:
            conv_str += f"{context['role']}: {context['content']} "
            
        conv_embed = annotate(conv_str)['data'][0]['embedding']
        conv_embed = np.asarray(conv_embed).reshape(1, -1)
        
        sim_mat = cosine_similarity(conv_embed, self.item_emb_arr)
        rank_arr = np.argsort(sim_mat, axis=-1).tolist()
        rank_arr = np.flip(rank_arr, axis=-1)[:, :50]
        item_rank_arr = self.id2item_id_arr[rank_arr].tolist()
        item_rank_arr = [[self.id2entityid[item_id] for item_id in item_rank_arr[0]]]
        
        return item_rank_arr, rec_labels
    
    def get_conv(self, conv_dict):
        
        context = conv_dict['context']
        context_list = [] # for model
        context_list.append({
            'role': 'system',
            'content': self.chat_recommender_instruction
        })
        
        for i, text in enumerate(context):
            if len(text) == 0:
                continue
            if i % 2 == 0:
                role_str = 'user'
            else:
                role_str = 'assistant'
            context_list.append({
                'role': role_str,
                'content': text
            })
        
        gen_inputs = None
        gen_str = annotate_chat(context_list)
        
        return gen_inputs, gen_str
    
    def get_choice(self, gen_inputs, options, state, conv_dict):
        
        updated_options = []
        for i, st in enumerate(state):
            if st >= 0:
                updated_options.append(options[i])
        
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        logit_bias = {encoding.encode(option)[0]: 10 for option in updated_options}
        
        context = conv_dict['context']
        context_list = [] # for model
        
        for i, text in enumerate(context[:-1]):
            if len(text) == 0:
                continue
            if i % 2 == 0:
                role_str = 'user'
            else:
                role_str = 'assistant'
            context_list.append({
                'role': role_str,
                'content': text
            })
        context_list.append({
            'role': 'user',
            'content': context[-1]
        })
        
        response_op = annotate_chat(context_list, logit_bias=logit_bias)
        return response_op[0]