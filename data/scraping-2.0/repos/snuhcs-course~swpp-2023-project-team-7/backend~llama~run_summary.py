import sys
from os import path
import uuid
import pickle
import openai
import tiktoken

from llama.constants import GPT_SYSTEM_SUMMARY_PROMPT_FROM_INTERMEDIATE, GPT_SYSTEM_SUMMARY_PROMPT_FROM_RAW

tokenizer = tiktoken.get_encoding("cl100k_base")

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def completion_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)

#TODO: handle python imports better
base_path = path.dirname(path.realpath(__file__))
sys.path.append(path.abspath(base_path))

def get_summary_from_text(progress, book_content_url):
    with open(book_content_url, 'r') as book_file:
        book_content = book_file.read()
    word_index = int(progress * len(book_content))
    read_content = book_content[:word_index]

    for resp in completion_with_backoff(
        model="gpt-4", messages=[
            {"role": "system", "content": GPT_SYSTEM_SUMMARY_PROMPT_FROM_RAW},
            {"role": "user", "content": read_content}
        ], stream=True
    ):
        finished = resp.choices[0].finish_reason is not None
        delta_content = "\n" if (finished) else resp.choices[0].delta.content
        sys.stdout.write(delta_content)
        sys.stdout.flush()

        yield delta_content, finished

        if finished:
            break


def get_summary_from_intermediate(progress, book_content_url, summary_tree_url):
    """
    generates summary based on the word_index
    :param progress: progress of the book
    :param book_id: book id to generate quiz from
    :param callback: callback function to call when a delta content is generated
    """

    summary_tree = ""
    with open(book_content_url, 'r') as book_file:
        book_content = book_file.read()
    with open(summary_tree_url, 'rb') as pickle_file:
        summary_tree = pickle.load(pickle_file)

    # word_index -> the number of characters read by the user.
    # start_index, end_idx is the number of tokens processed by the summary
    word_index = int(progress * len(book_content))
    read_content = book_content[:word_index]
    tokenized_read_content = tokenizer.encode(read_content)
    word_index = len(tokenized_read_content) - 1

    leaf = summary_tree.find_leaf_summary(word_index=word_index)
    available_summary_list = summary_tree.find_included_summaries(leaf)

    content = "\n\n".join([summary.summary_content for summary in available_summary_list])
    content += "\n\n" + book_content[leaf.start_idx:word_index]

    for resp in completion_with_backoff(
        model="gpt-4", messages=[
            {"role": "system", "content": GPT_SYSTEM_SUMMARY_PROMPT_FROM_INTERMEDIATE},
            {"role": "user", "content": content}
        ], stream=True
    ):
        finished = resp.choices[0].finish_reason is not None
        delta_content = "\n" if (finished) else resp.choices[0].delta.content
        sys.stdout.write(delta_content)
        sys.stdout.flush()

        yield delta_content, finished

        if finished:
            break

if __name__ == "__main__":
    get_summary_from_intermediate(10880, 1)