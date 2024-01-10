# Copyright 2023 Boris Zubarev. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict, List, Optional

import openai
from loguru import logger

from wgpt import enums
from wgpt.core.prompts import ASSISTANT_PROMPT


class GPTClient:
    DEFAULT_MODEL_NAME: str = "gpt-3.5-turbo-16k"

    def __init__(
            self,
            num_completion: int = 1,
            temperature: float = 1.0,
            max_tokens: int = 256,
            top_p: float = 0.99,
            frequency_penalty: float = 0.0,
            presence_penalty: float = 0.0,
    ):
        self.num_completion = num_completion
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty

    def get_gpt_response(
            self,
            messages: List[Dict[str, str]],
            model_name: Optional[str] = None,
            num_completion: Optional[int] = None,
            num_retries: int = 3,
    ) -> List[str]:
        model_name = model_name or self.DEFAULT_MODEL_NAME
        num_completion = num_completion or self.num_completion

        retries_counter = 0
        text_responses: List[str] = list()

        while True:
            if retries_counter >= num_retries:
                break

            try:
                open_ai_response = openai.ChatCompletion.create(
                    model=model_name,
                    messages=messages,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_tokens=self.max_tokens,
                    presence_penalty=self.presence_penalty,
                    frequency_penalty=self.frequency_penalty,
                    n=num_completion,
                )
                text_responses = [choice.message.content for choice in open_ai_response.choices]
                break
            except Exception as exception:
                logger.error(f"GPT response exception: {exception}")
                retries_counter += 1

        return text_responses

    def one_turn_generation(
            self,
            content: str,
            assistant_prompt: Optional[str] = None,
            model_name: Optional[str] = None,
            num_completion: Optional[int] = None,
    ) -> List[str]:
        assistant_prompt = assistant_prompt or ASSISTANT_PROMPT

        messages = [
            {enums.Field.role: enums.GPTRole.system, enums.Field.content: assistant_prompt},
            {
                enums.Field.role: enums.GPTRole.user,
                enums.Field.content: content,
            },
        ]
        text_responses = self.get_gpt_response(messages=messages, model_name=model_name, num_completion=num_completion)
        return text_responses
