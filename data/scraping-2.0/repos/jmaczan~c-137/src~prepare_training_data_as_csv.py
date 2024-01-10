from kaggle.api.kaggle_api_extended import KaggleApi
import pandas as pd
from langchain import PromptTemplate
from os import path

dataset_csv_filename = "RickAndMortyScripts.csv"
dataset_name = "andradaolteanu/rickmorty-scripts"
expected_data_size = 1905

if not path.exists(dataset_csv_filename):
    api = KaggleApi()
    api.authenticate()

    api.dataset_download_file(dataset_name, dataset_csv_filename)

data = pd.read_csv(dataset_csv_filename)

assert len(data.index) == expected_data_size

prompt_template = PromptTemplate.from_template(
    "<s>[INST] {other_lines} [/INST] {rick_lines} </s>"
)


def build_data_item(other_lines, rick_lines):
    return prompt_template.format(other_lines=other_lines, rick_lines=rick_lines)


def clear_unwanted_characters(text):
    return text.replace('"', "")


def preprocess_data_item(text):
    return clear_unwanted_characters(text) + " "


rick = "Rick"


global_output = pd.DataFrame(columns=["text"])


global_pointer = -1
other_lines = ""
rick_lines = ""
total_lines_count = expected_data_size

local_pointer = 0

while global_pointer < total_lines_count:
    global_pointer += local_pointer

    if global_pointer == total_lines_count:
        break

    local_pointer = 0
    other_lines = ""
    rick_lines = ""

    current_row_index = global_pointer + local_pointer

    if current_row_index >= total_lines_count:
        break

    row = data.iloc[current_row_index]

    while rick_lines == "" or rick in row["name"]:
        preprocessed_line = preprocess_data_item(row["line"])
        if row["name"] == rick:
            rick_lines += preprocessed_line
        else:
            other_lines += preprocessed_line

        local_pointer += 1
        current_row_index = global_pointer + local_pointer
        if current_row_index >= total_lines_count:
            break
        row = data.iloc[current_row_index]

    global_output = pd.concat(
        [global_output, pd.DataFrame([build_data_item(other_lines, rick_lines)])],
        ignore_index=True,
    )

global_output.to_csv("dataset.csv", index=False)
