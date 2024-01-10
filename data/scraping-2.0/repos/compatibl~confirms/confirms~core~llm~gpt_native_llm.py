# Copyright (C) 2023-present The Project Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from dataclasses import dataclass, field
from typing import Dict, Optional

import openai

from confirms.core.llm.llm import Llm
from confirms.core.settings import Settings


@dataclass
class GptNativeLlm(Llm):
    """GPT model family using native OpenAI API."""

    temperature: float = field(default=None)
    """Model temperature (note that for GPT models zero value does not mean reproducible answers)."""

    _llm: bool = field(default=None)

    def load_model(self):
        """Load model after fields have been set."""

        # Skip if already loaded
        if self._llm is None:
            gpt_model_types = ["gpt-3.5-turbo", "gpt-4"]
            if self.model_type not in gpt_model_types:
                raise RuntimeError(
                    f"GPT Native LLM model type {self.model_type} is not recognized. "
                    f"Valid model types are {gpt_model_types}"
                )

            # Native OpenAI API calls are stateless. This means no object is needed at this time.
            self._llm = True

    def completion(self, question: str, *, prompt: Optional[str] = None) -> str:
        """Simple completion with optional prompt."""

        # Load settings
        Settings.load()

        if prompt is not None:
            messages = [{"role": "system", "content": prompt}]
        else:
            messages = []

        messages = messages + [{"role": "user", "content": question}]

        response = openai.ChatCompletion.create(model=self.model_type, messages=messages)
        answer = response['choices'][0]['message']['content']
        return answer

    def function_completion(self, question: str, *, prompt: Optional[str] = None) -> Dict[str, str]:
        """Completion with functions."""

        # Load settings
        Settings.load()

        if prompt is not None:
            messages = [{"role": "system", "content": prompt}]
        else:
            messages = []

        messages = messages + [{"role": "user", "content": question}]
        functions = [
            {
                "name": "get_interest_schedule",
                "description": "Calculates and returns interest schedule from function parameters",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "first_unadjusted_payment_date": {
                            "type": "string",
                            "description": "First unadjusted payment date using ISO 8601 date format yyyy-mm-dd.",
                        },
                        "last_unadjusted_payment_date": {
                            "type": "string",
                            "description": "Last unadjusted payment date using ISO 8601 date format yyyy-mm-dd.",
                        },
                        "payment_frequency": {
                            "type": "string",
                            "description": "Payment frequency expressed as the number of months followed by capital M",
                            "enum": ["1M", "3M", "6M", "12M"],
                        },
                    },
                    "required": ["first_unadjusted_payment_date", "last_unadjusted_payment_date", "payment_frequency"],
                },
            },
            {
                "name": "get_payment_frequency",
                "description": "Extract payment frequency from description",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "payment_frequency": {
                            "type": "string",
                            "description": "Payment frequency expressed as one word",
                        },
                    },
                    "required": ["payment_frequency"],
                },
            },
        ]
        response = openai.ChatCompletion.create(
            model=self.model_type,
            messages=messages,
            functions=functions,
            function_call="auto",  # auto is default, but we'll be explicit
        )
        response_message = response["choices"][0]["message"]

        if response_message.get("function_call"):
            function_name = response_message["function_call"]["name"]
            result = json.loads(response_message["function_call"]["arguments"])
            result["function"] = function_name
            return result
        else:
            raise RuntimeError("No functions called in response to message.")
