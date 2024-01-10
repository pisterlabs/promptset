# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""
This module provides functions for asking questions about a dataframe.

"""
import os
from functools import partial
from pathlib import Path
from typing import List, Optional, Tuple, Union

import openai
import yaml

import pandas as pd
from IPython.display import Markdown
import msticpy as mp


from _version import VERSION

__version__ = VERSION
__author__ = "Ian Hellen"

openai.api_key  = os.getenv('OPENAI_API_KEY')


class PDQuery:

    def __init__(self, prompt_paths: Union[str, List[str]] = None, model: str = "gpt-3.5-turbo"):
        self._model = model
        self._prompt_paths = prompt_paths
        self._prompt_dict = self._load_prompts()
        self._prompt = self._prompt_dict["default"]

    def _load_prompts(self):
        if isinstance(self._prompt_paths, str):
            self._prompt_paths = [self._prompt_paths]
        prompt_dict = {}
        for path in self._prompt_paths:
            prompt_dict.update(yaml.safe_load(Path(path).read_text(encoding="utf-8")))
        return prompt_dict

    def _create_prompt_partials(self):
        self._prompt_partials = {}
        df_prompt = self._prompt_dict["df_generic_task"].get("prompt")
        for name, prompt_def in self._prompt_dict.items():
            self._prompt_partials[key] = partial(analyze_df, prompt=df_prompt, task=prompt_def["task"])




def get_completion(prompt, model="gpt-3.5-turbo", max_tokens=2000, temperature=0):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message["content"]


def analyze_df(data: pd.DataFrame, prompt: str, task: str, sample_rows: int = 5, max_tokens: int = 1000):
    """
    Analyze a pandas dataframe and return a JSON dictionary with the results.

    Parameters
    ----------
    data : pd.DataFrame
        The dataframe to analyze.
    sample_rows : int, optional
        The number of rows to display in the prompt, by default 5

    Returns
    -------
    dict

    """
    df_head = data.head(sample_rows)
    num_rows = len(data)
    num_columns = len(data.columns)
    return get_completion(
        prompt.format(
            df_head=df_head,
            rows_to_display=sample_rows,
            num_rows=num_rows,
            num_columns=num_columns,
            task=task,
        ),
        max_tokens=max_tokens,
    )

