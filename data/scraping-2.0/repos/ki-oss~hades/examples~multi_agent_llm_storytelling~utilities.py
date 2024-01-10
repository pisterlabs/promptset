# Copyright 2023 Brit Group Services Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import os
from dataclasses import asdict

import openai
from gpt_json import GPTMessage, GPTModelVersion
from openai.error import APIError, RateLimitError, Timeout

API_KEY = os.environ["OPENAI_API_KEY"]


async def plaintext_chat_response(messages: list[GPTMessage], tries=7):
    try:
        plaintext_response = await openai.ChatCompletion.acreate(
            api_key=API_KEY, model=GPTModelVersion.GPT_3_5, messages=[asdict(message) for message in messages]
        )
        return plaintext_response.choices[0].message.content
    except (RateLimitError, APIError, Timeout) as e:
        if tries == 0:
            raise e
        await asyncio.sleep((8 - tries) ** 2)
        return await plaintext_chat_response(messages, tries - 1)
