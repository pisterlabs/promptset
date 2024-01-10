#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# scenario_functions.py
#   functions used in sample Japanese app
#   日本語サンプルアプリで用いるシナリオ関数

__version__ = '0.1'
__author__ = 'Mikio Nakano'
__copyright__ = 'C4A Research Institute, Inc.'

from datetime import datetime
from typing import Dict, Any
import os

use_openai: bool = False

openai_key: str = os.environ.get('OPENAI_KEY', "")
if openai_key:
    import openai
    use_openai = True
    openai.api_key = openai_key


# 知っているラーメンの種類
known_ramens = ("豚骨ラーメン", "味噌ラーメン", "塩ラーメン", "醤油ラーメン")

def is_known_ramen(ramen: str, context: Dict[str, Any]) -> bool:
    """
    知っているラーメンかどうか
    :param ramen: ラーメンの種類名
    :param context: 対話文脈（未使用）
    :return: 知っていたらTrue, さもなくばFalse
    """
    return ramen in known_ramens


def is_novel_ramen(ramen: str, context: Dict[str, Any]) -> bool:
    """
    知らないラーメンかどうか
    :param ramen: ラーメンの種類名
    :param context: 対話文脈（未使用）
    :return: 知らないならTrue, 知っていればFalse
    """
    return ramen not in known_ramens


# ラーメンの種類と地域の関係
ramen_map = {"豚骨ラーメン": "博多",
             "味噌ラーメン": "札幌",
             "塩ラーメン": "函館",
             "醤油ラーメン": "東京"}


def get_ramen_location(ramen: str, variable: str, context: Dict[str, Any]) -> None:
    """
    ラーメンの種類からその発祥の地域を得て、対話文脈に保持する
    :param ramen: ラーメンの種類
    :param variable: 地域名を蓄える変数の名前
    :param context: 対話文脈
    """
    location:str = ramen_map.get(ramen, "日本")
    context[variable] = location


def decide_greeting(greeting_variable: str, context: Dict[str, Any]) -> None:
    """
    挨拶文を時間帯に応じて決める
    :param greeting_variable: 挨拶を表す変数の名前
    :param context: 対話文脈
    """

    hour: int = datetime.now().hour
    if hour < 4:
        context[greeting_variable] = "こんばんは"
    elif hour < 10:
        context[greeting_variable] = "おはようございます"
    elif hour <= 19:
        context[greeting_variable] = "こんにちは"
    else:
        context[greeting_variable] = "こんばんは"


def generate_with_openai_gpt(prompt: str):

    chat_completion = None
    while True:
        try:
            chat_completion = self._openai_client.with_options(timeout=10).chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
            )
        except openai.APITimeoutError:
            continue
        except Exception as e:
            self.log_error("OpenAI Error: " + traceback.format_exc())
            sys.exit(1)
        finally:
            if not chat_completion:
                continue
            else:
                break
    generated_utterance: str = chat_completion.choices[0].message.content
    return generated_utterance


def set_impression_of_dialogue(impression_key: str, context: Dict[str, Any]) -> None:

    if use_openai:

        prompt = ""
        for turn in context["_dialogue_history"]:
            if turn["speaker"] == 'user':
                prompt += f"ユーザ「{turn['utterance']}」\n"
            else:
                prompt += f"システム「{turn['utterance']}」\n"
        prompt += "の後、システムが感想を短く言う発話を生成してください。"

        generated_utterance: str = generate_with_openai_gpt(prompt)
        impression = generated_utterance.replace("システム「", "").replace("「", "").replace("」", "")

    else:
        impression = "そうなんですね"

    context[impression_key] = impression


def generate_confirmation_request(nlu_result: Dict[str, Any], context: Dict[str, Any]) -> str:

    if nlu_result.get("type") == "特定のラーメンが好き" and nlu_result["slots"].get("favorite_ramen"):
        return f'{nlu_result["slots"]["favorite_ramen"]}がお好きなんですか？'
    else:
        return "もう一度言って頂けますか？"


