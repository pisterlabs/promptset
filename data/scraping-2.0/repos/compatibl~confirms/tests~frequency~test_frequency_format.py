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
from confirms.core.llm.gpt_native_llm import GptNativeLlm
from confirms.core.llm.llama_lang_chain_llm import LlamaLangChainLlm

SIMPLE_CONTEXT = (
    '```Effective Date: 15 June 2010.'
    'Maturity Date: 15 June 2020.'
    'Payment Frequency: Quarterly.```'
)

EXPLICIT_CONTEXT = (
    '```Maturity Date: 18-July-2033. '
    'Interest Payment Date: Interest payments shall be made quarterly on each 18th day of the months of '
    'April, July, October and January, commencing October 18, 2023, and ending on the Maturity Date.```'
)

IMPLICIT_CONTEXT = (
    '```Issue Date: 9 July 2009 (Settlement Date). '
    'Maturity Date: 9 July 2013.'
    'Interest Payment Dates: The 9th of each January, April, July, and October '
    'commencing 9 October 2009 with a final payment on the Maturity Date.```'
)

VERBOSE_CONTEXT = (
    '```Issue Date: On or about December 27, 2013.'
    'Maturity Date and Term: On or about December 27, 2023, resulting in a term to maturity '
    'of approximately 10 years. The $100 principal amount (the "Principal '
    'Amount") will only be payable at maturity. For further information, see "Payments under the Notes". '
    'Interest Payment Date: The first Interest payment, if any, shall be made on June 27, 2014, following '
    'which Holders of the Notes will be entitled to receive semi-annual Interest payments, if any. '
    'Subject to the occurrence of certain Extraordinary Events, Interest, if any, will be payable on the '
    '27th day of June and December of each year that the Notes remain outstanding (each, an "Interest Payment'
    ' Date") from and including June 27, 2014 to and including the Maturity Date. If any Interest Payment '
    'Date is not a Business Day, it will be postponed to the next following Business Day.```'
)

PLAIN_TEMPLATE = (
    "<s>[INST] Pay attention and remember information below, " 
    "which will help to answer the question or imperative after the context ends. " 
    "Context: {context}. " 
    "According to only the information in the document sources provided within the context above, output payment "
    "frequency only as one word and do not include any other text in your response. The payment frequency is[/INST]"
)

ONE_WORD_TEMPLATE = (
    "<s>[INST] Pay attention and remember information below, " 
    "which will help to answer the question or imperative after the context ends. " 
    "Context: {context}. " 
    "According to only the information in the document sources provided within the context above, "
    "the payment frequency is[/INST]"
)


def run_frequency_extraction(*, template: str, context: str, result_name: str, temperature: Optional[float] = None):
    """Function completion for payment frequency extraction."""

    results = []
    for seed in range(1, 26):
        cur_result = {}
        model_types = ["gpt-3.5-turbo", "gpt-4", "llama-2-7b-chat.Q4_K_M.gguf", "llama-2-13b-chat.Q4_K_M.gguf"]
        # , "llama-2-70b-chat.Q4_K_M.gguf"]
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


def test_frequency_extraction_plain_simple():
    """Test function completion for payment frequency extraction with default model settings
    for plain template and simple context."""
    run_frequency_extraction(template=PLAIN_TEMPLATE, context=SIMPLE_CONTEXT, result_name="frequency_plain_simple")


def test_frequency_extraction_plain_explicit():
    """Test function completion for payment frequency extraction with default model settings
    for plain template and explicit context."""
    run_frequency_extraction(template=PLAIN_TEMPLATE, context=EXPLICIT_CONTEXT, result_name="frequency_plain_explicit")


def test_frequency_extraction_plain_implicit():
    """Test function completion for payment frequency extraction with default model settings
    for plain template and implicit context."""
    run_frequency_extraction(template=PLAIN_TEMPLATE, context=IMPLICIT_CONTEXT, result_name="frequency_plain_implicit")


def test_frequency_extraction_plain_verbose():
    """Test function completion for payment frequency extraction with default model settings
    for plain template and implicit context."""
    run_frequency_extraction(template=PLAIN_TEMPLATE, context=VERBOSE_CONTEXT, result_name="frequency_plain_verbose")


def test_frequency_extraction_word_simple():
    """Test function completion for payment frequency extraction with default model settings
    for one word template and simple context."""
    run_frequency_extraction(template=ONE_WORD_TEMPLATE, context=SIMPLE_CONTEXT, result_name="frequency_word_simple")


def test_frequency_extraction_word_explicit():
    """Test function completion for payment frequency extraction with default model settings
    for one word template and explicit context."""
    run_frequency_extraction(template=ONE_WORD_TEMPLATE, context=EXPLICIT_CONTEXT, result_name="frequency_word_explicit")


def test_frequency_extraction_word_implicit():
    """Test function completion for payment frequency extraction with default model settings
    for one word template and implicit context."""
    run_frequency_extraction(template=ONE_WORD_TEMPLATE, context=IMPLICIT_CONTEXT, result_name="frequency_word_implicit")


def test_frequency_extraction_word_verbose():
    """Test function completion for payment frequency extraction with default model settings
    for one word template and implicit context."""
    run_frequency_extraction(template=ONE_WORD_TEMPLATE, context=VERBOSE_CONTEXT, result_name="frequency_word_verbose")


def test_logit_processing():
    """Test function completion with input from previous step."""

    results_dir = os.path.join(os.path.dirname(__file__), "../../results")
    file_path = os.path.join(results_dir, "frequency_explicit.csv")

    df = pd.read_csv(file_path)
    input_data = df['llama-2-7b-chat.Q4_K_M.gguf']

    results = [input_data]
    model_types = ["gpt-3.5-turbo", "gpt-4", "llama-2-7b-chat.Q4_K_M.gguf", "llama-2-13b-chat.Q4_K_M.gguf"]
    for model_type in model_types:
        current_result = []
        if model_type.startswith("llama"):
            llm = LlamaLangChainLlm(model_type=model_type, grammar_file="frequency_word.gbnf")
        elif model_type.startswith("gpt"):
            llm = GptNativeLlm(model_type=model_type, temperature=0.0)
        else:
            raise RuntimeError(f"Unknown model type: {model_type}")
        for context in input_data:
            question = (
                "<s>[INST] Pay attention and remember information below, "
                "which will help to answer the question or imperative after the context ends. "
                f"Context: {context}. "
                "According to only the information in the document sources provided within the context above, "
                "the payment frequency is [/INST]"
            )
            if model_type.startswith("llama"):
                answer = llm.completion(question)
                current_result.append(answer)
            elif model_type.startswith("gpt"):
                answer = llm.function_completion(question)
                current_result.append(answer['payment_frequency'])
        results.append(pd.DataFrame(current_result))

    outputs_dir = os.path.join(os.path.dirname(__file__), "../../results")
    output_path = os.path.join(outputs_dir, "frequency_logit_processing.csv")

    df = pd.concat(results, axis=1)
    df.columns = ['input'] + model_types
    df.to_csv(output_path, index=False)


if __name__ == '__main__':
    pytest.main([__file__])
