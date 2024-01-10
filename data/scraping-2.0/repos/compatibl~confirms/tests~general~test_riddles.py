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

import os
from typing import Optional

import pandas as pd
import pytest
from langchain import PromptTemplate

from confirms.core.llm.gpt_lang_chain_llm import GptLangChainLlm
from confirms.core.llm.llama_lang_chain_llm import LlamaLangChainLlm


def run_sally_riddle(*, result_name: str, temperature: Optional[float] = None):
    """Run for solving Sally and her siblings riddle."""

    context = "```Sally has three brothers. Each of Sally's brothers has two sisters.```"
    template = (
        "<s>[INST] Pay attention and remember information below, which will help to answer the question "
        "or imperative after the context ends. "
        "Context: {context}. "
        "According to only the information in the document sources provided within the context above, "
        "how many sisters does Sally have? [/INST]"
    )

    results = []
    for seed in range(1, 26):
        cur_result = {}
        model_types = ["gpt-3.5-turbo", "gpt-4", "llama-2-7b-chat.Q4_K_M.gguf", "llama-2-13b-chat.Q4_K_M.gguf"]
        for model_type in model_types:
            if model_type.startswith("llama"):
                llm = LlamaLangChainLlm(model_type=model_type, temperature=temperature, seed=seed)
            elif model_type.startswith("gpt"):
                llm = GptLangChainLlm(model_type=model_type, temperature=temperature)
            else:
                raise RuntimeError(f"Unknown model type: {model_type}")

            prompt = PromptTemplate(template=template, input_variables=["context"])
            answer = llm.completion(context, prompt=prompt)
            cur_result[model_type] = answer
        results.append(pd.DataFrame([cur_result]))

    outputs_dir = os.path.join(os.path.dirname(__file__), "../../results")
    output_path = os.path.join(outputs_dir, f"{result_name}.csv")

    df = pd.concat(results)
    df.to_csv(output_path, index=False)


def run_apples_riddle(*, result_name: str, temperature: Optional[float] = None):
    """Test for solving apples in a box riddle."""

    context = (
        "```A green apple is in the same box as three red apples. "
        "Each of these three red apples is in the same box as two green apples.```"
    )
    template = (
        "<s>[INST] Pay attention and remember information below, "
        "which will help to answer the question or imperative after the context ends. "
        "Context: {context}. "
        "According to only the information in the document sources provided within the context above, "
        "how many other green apples are in the same box as the original green apple? [/INST]"
    )

    results = []
    for seed in range(1, 26):
        cur_result = {}
        model_types = ["gpt-3.5-turbo", "gpt-4", "llama-2-7b-chat.Q4_K_M.gguf", "llama-2-13b-chat.Q4_K_M.gguf"]
        for model_type in model_types:
            if model_type.startswith("llama"):
                llm = LlamaLangChainLlm(model_type=model_type, temperature=temperature, seed=seed)
            elif model_type.startswith("gpt"):
                llm = GptLangChainLlm(model_type=model_type, temperature=temperature)
            else:
                raise RuntimeError(f"Unknown model type: {model_type}")

            prompt = PromptTemplate(template=template, input_variables=["context"])
            answer = llm.completion(context, prompt=prompt)
            cur_result[model_type] = answer
        results.append(pd.DataFrame([cur_result]))

    outputs_dir = os.path.join(os.path.dirname(__file__), "../../results")
    output_path = os.path.join(outputs_dir, f"{result_name}.csv")

    df = pd.concat(results)
    df.to_csv(output_path, index=False)


def test_sally_riddle():
    """Test for solving Sally and her siblings riddle with default model settings."""
    run_sally_riddle(result_name="sally_riddle")


def test_sally_riddle_temp08():
    """Test for solving Sally and her siblings riddle with temperature=0.8."""
    run_sally_riddle(result_name="sally_riddle_temp08", temperature=0.8)


def test_apples_riddle():
    """Test for solving apples in a box riddle with default model settings."""
    run_apples_riddle(result_name="apples_riddle")


def test_apples_riddle_temp08():
    """Test for solving apples in a box riddle with temperature=0.8."""
    run_apples_riddle(result_name="apples_riddle_temp08", temperature=0.8)


if __name__ == '__main__':
    pytest.main([__file__])
