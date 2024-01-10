#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2023 The community Authors.
# A-Tune is licensed under the Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#     http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR
# PURPOSE.
# See the Mulan PSL v2 for more details.
# Create: 2023/10/10

import json
import os
import openai
from pilot.configs.model_config import config_parser

openai.api_key = config_parser.get('gpt', 'openai_key')

messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Who won the world series in 2020?"},
        {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
        {"role": "user", "content": "Where was it played?"}
    ]


def chat_gpt(messages, model="gpt-3.5-turbo", stream=False):
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        stream=stream
    )
    response = response['choices'][0]['message']['content']
    return response


def chat_gpt_stream(messages, model="gpt-3.5-turbo", stream=True):
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        stream=stream
    )
    for chunk in response:
        content = ''
        if "content" in chunk["choices"][0]["delta"]:
            content = chunk["choices"][0]["delta"]["content"]
            data = json.dumps({"answer": content}, ensure_ascii=False)
            yield f"data: {data}\n"