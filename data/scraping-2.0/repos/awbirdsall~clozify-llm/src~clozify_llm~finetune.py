"""finetune.py Functionality to finetune completion LLM with training data
"""
import json
from pathlib import Path

import openai
import pandas as pd

from clozify_llm.constants import (
    CLOZE_COL,
    DEFAULT_COMPLETION_MODEL,
    DEFN_COL,
    WORD_COL,
)
from clozify_llm.utils import format_completion, format_prompt


class FineTuner:
    def __init__(self, df: pd.DataFrame, training_data_path: str, model: str = DEFAULT_COMPLETION_MODEL):
        self.df = df
        self.training_data_path = training_data_path
        self.model = model

    def create_dataset(self) -> list[dict]:
        train_rows = []
        to_iter = self.df[[WORD_COL, DEFN_COL, "text", "translation", CLOZE_COL]]
        for row in to_iter.itertuples():
            prompt = format_prompt(getattr(row, WORD_COL), getattr(row, DEFN_COL))
            completion = format_completion(row.text, row.translation, getattr(row, CLOZE_COL))
            train_rows.append({"prompt": prompt, "completion": completion})
        return train_rows

    def write_data(self, dataset, overwrite=False):
        """Write dataset to local training_data_path.

        Do not overwrite unless instructed"""
        if Path(self.training_data_path).exists() and not overwrite:
            print(f"{self.training_data_path} exists, skipping write")
        else:
            with open(self.training_data_path, "w", encoding="utf-8") as f:
                for entry in dataset:
                    json.dump(entry, f, ensure_ascii=False)
                    f.write("\n")

    def start_finetuning(self, generate_dataset=True):
        """Start finetuning and return response from openai.FineTune.create

        Note the fine tune will only be submitted when this call completes. To follow status can use
        openai CLI commands such as

        ```bash
        $ openai api fine_tunes.follow -i <FINE_TUNE_ID>
        ```

        where <FINE_TUNE_ID> is the "id" field in the fine_tune.create response.
        """
        if generate_dataset:
            dataset = self.create_dataset()
            self.write_data(dataset, overwrite=True)

        with open(self.training_data_path, "r") as f:
            file_response = openai.File.create(file=f, purpose="fine-tune")
        print(f"File.create with id {file_response.id}")

        ft_response = openai.FineTune.create(training_file=file_response.id, model=self.model)
        return ft_response
