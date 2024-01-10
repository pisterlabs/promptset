import sys
from os import path
import pickle
import openai
import tiktoken

from llama.constants import GPT_SYSTEM_QUIZ_PROMPT_FROM_INTERMEDIATE, GPT_SYSTEM_QUIZ_PROMPT_FROM_RAW

tokenizer = tiktoken.get_encoding("cl100k_base")

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def completion_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)

def get_quizzes_from_text(progress, book_content_url):
    """
    generates 10 quizzes based on the word_index
    :param progress: progress of the book
    :param book_id: book id to generate quiz from
    """
    with open(book_content_url, 'r') as book_file:
        book_content = book_file.read()

    # word_index -> the number of characters read by the user.
    # start_index, end_idx is the number of tokens processed by the summary
    word_index = int(progress * len(book_content))
    read_content = book_content[:word_index]

    for resp in completion_with_backoff(
        model="gpt-4", messages=[
            {"role": "system", "content": GPT_SYSTEM_QUIZ_PROMPT_FROM_RAW},
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

def get_quizzes_from_intermediate(progress, book_content_url, summary_tree_url):
    """
    generates 10 quizzes based on the word_index
    :param progress: progress of the book
    :param book_id: book id to generate quiz from
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
    word_index = len(tokenized_read_content)-1
    
    # generate new quiz
    leaf = summary_tree.find_leaf_summary(word_index=word_index)
    available_summary_list = summary_tree.find_included_summaries(leaf)

    content = "\n\n".join([summary.summary_content for summary in available_summary_list])
    content += "\n\n" + book_content[leaf.end_idx:word_index]

    for resp in completion_with_backoff(
        model="gpt-4", messages=[
            {"role": "system", "content": GPT_SYSTEM_QUIZ_PROMPT_FROM_INTERMEDIATE},
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
    # main()
    # key = None
    # for quiz, quiz_len in get_quizzes(progress=10880 / len(book_content), book_id=1):
    #     print(quiz, quiz_len)
    pass