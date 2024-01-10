import json
import os
import random
import typing
from argparse import ArgumentParser

import openai
from loguru import logger
from tenacity import _utils, Retrying, retry_if_not_exception_type
from tenacity.stop import stop_base
from tenacity.wait import wait_base


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


def annotate(item_text_list):
    request_timeout = 6
    for attempt in Retrying(
        reraise=True, retry=retry_if_not_exception_type((openai.error.InvalidRequestError, openai.error.AuthenticationError)),
        wait=my_wait_exponential(min=1, max=60), stop=(my_stop_after_attempt(8)), before_sleep=my_before_sleep
    ):
        with attempt:
            response = openai.Embedding.create(
                model='text-embedding-ada-002', input=item_text_list, request_timeout=request_timeout
            )
        request_timeout = min(30, request_timeout * 2)

    return response


def get_exist_item_set():
    exist_item_set = set()
    for file in os.listdir(save_dir):
        user_id = os.path.splitext(file)[0]
        exist_item_set.add(user_id)
    return exist_item_set


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--api_key')
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--dataset', type=str, choices=['redial', 'opendialkg'])
    args = parser.parse_args()

    openai.api_key = args.api_key
    batch_size = args.batch_size
    dataset = args.dataset

    save_dir = f'../save/embed/item/{dataset}'
    os.makedirs(save_dir, exist_ok=True)

    with open(f'../data/{dataset}/id2info.json', encoding='utf-8') as f:
        id2info = json.load(f)

    # redial
    if dataset == 'redial':
        info_list = list(id2info.values())
        item_texts = []
        for info in info_list:
            item_text_list = [
                f"Title: {info['name']}", f"Genre: {', '.join(info['genre']).lower()}",
                f"Star: {', '.join(info['star'])}",
                f"Director: {', '.join(info['director'])}", f"Plot: {info['plot']}"
            ]
            item_text = '; '.join(item_text_list)
            item_texts.append(item_text)
        attr_list = ['genre', 'star', 'director']

    # opendialkg
    if dataset == 'opendialkg':
        item_texts = []
        for info_dict in id2info.values():
            item_attr_list = [f'Name: {info_dict["name"]}']
            for attr, value_list in info_dict.items():
                if attr != 'title':
                    item_attr_list.append(f'{attr.capitalize()}: ' + ', '.join(value_list))
            item_text = '; '.join(item_attr_list)
            item_texts.append(item_text)
        attr_list = ['genre', 'actor', 'director', 'writer']

    id2text = {}
    for item_id, info_dict in id2info.items():
        attr_str_list = [f'Title: {info_dict["name"]}']
        for attr in attr_list:
            if attr not in info_dict:
                continue
            if isinstance(info_dict[attr], list):
                value_str = ', '.join(info_dict[attr])
            else:
                value_str = info_dict[attr]
            attr_str_list.append(f'{attr.capitalize()}: {value_str}')
        item_text = '; '.join(attr_str_list)
        id2text[item_id] = item_text

    item_ids = set(id2info.keys()) - get_exist_item_set()
    while len(item_ids) > 0:
        logger.info(len(item_ids))

        # redial
        if dataset == 'redial':
            batch_item_ids = random.sample(tuple(item_ids), min(batch_size, len(item_ids)))
            batch_texts = [id2text[item_id] for item_id in batch_item_ids]

        # opendialkg
        if dataset == 'opendialkg':
            batch_item_ids = random.sample(tuple(item_ids), min(batch_size, len(item_ids)))
            batch_texts = [id2text[item_id] for item_id in batch_item_ids]

        batch_embeds = annotate(batch_texts)['data']
        for embed in batch_embeds:
            item_id = batch_item_ids[embed['index']]
            with open(f'{save_dir}/{item_id}.json', 'w', encoding='utf-8') as f:
                json.dump(embed['embedding'], f, ensure_ascii=False)

        item_ids -= get_exist_item_set()
