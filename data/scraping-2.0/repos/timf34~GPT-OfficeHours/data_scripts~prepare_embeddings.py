import openai
import random


# Sources:
# - https://github.com/openai/openai-cookbook/blob/main/examples/Question_answering_using_embeddings.ipynb
# - https://github.com/EveryInc/transcriptbot/blob/main/data_preparation/prepare_transcript_embeddings.py
# - https://simonwillison.net/2023/Jan/13/semantic-search-answers/
# - https://github.com/jerryjliu/gpt_index


def get_number_of_words_in_file(file_path: str) -> int:
    with open(file_path, "r") as f:
        return len(f.read().split())


def get_number_of_lines_in_file(file_path: str) -> int:
    with open(file_path, "r") as f:
        return len(f.readlines())


def read_txt_file(file_path: str) -> str:
    with open(file_path, "r") as f:
        return f.read()


class PrepareEmbeddings:
    def __init__(self):
        pass

