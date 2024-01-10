import psycopg2
#import mwparserfromhell  # for splitting Wikipedia articles into sections
import openai  # for generating embeddings
import pandas as pd  # for DataFrames to store article sections and embeddings
import re  # for cutting <ref> links out of Wikipedia articles
import tiktoken  # for counting tokens
from lib.appsync import query_api
import os

GPT_MODEL = "gpt-3.5-turbo"  # only matters insofar as it selects which tokenizer to use
MAX_TOKENS = 1600
EMBEDDING_MODEL = "text-embedding-ada-002"  # OpenAI's best embeddings as of Apr 2023

openai.api_key = os.environ.get('OPENAPI_KEY')

def insert_row(id):
    conn = psycopg2.connect(os.environ.get('PG_DSN'))
    document = get_document(id)
    text = clean_text(document.get('content'))
    text_strings = []
    text_strings.extend(split_strings_from_text(text, max_tokens=MAX_TOKENS))

    print(f"{len(text)} charachters split into {len(text_strings)} strings.")
    response = openai.Embedding.create(input=text_strings, model='text-embedding-ada-002')
    embeddings = [v['embedding'] for v in response['data']]
    with conn.cursor() as cursor:
        for content, embedding in zip(text_strings, embeddings):
            cursor.execute('INSERT INTO documents (id, content, embedding) VALUES (%s, %s, %s)', (id, content, embedding))
    conn.commit()
    cursor.close()
    conn.close()
    return id

# clean text
def clean_text(text: str):
    """
    Return a cleaned up section with:
        - <ref>xyz</ref> patterns removed
        - leading/trailing whitespace removed
    """
    text = re.sub(r"<ref.*?</ref>", "", text)
    text = text.strip()
    return text

def num_tokens(text: str, model: str = GPT_MODEL):
    """Return the number of tokens in a string."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


def halved_by_delimiter(string: str, delimiter: str = "\n"):
    """Split a string in two, on a delimiter, trying to balance tokens on each side."""
    chunks = string.split(delimiter)
    if len(chunks) == 1:
        return [string, ""]  # no delimiter found
    elif len(chunks) == 2:
        return chunks  # no need to search for halfway point
    else:
        total_tokens = num_tokens(string)
        halfway = total_tokens // 2
        best_diff = halfway
        for i, chunk in enumerate(chunks):
            left = delimiter.join(chunks[: i + 1])
            left_tokens = num_tokens(left)
            diff = abs(halfway - left_tokens)
            if diff >= best_diff:
                break
            else:
                best_diff = diff
        left = delimiter.join(chunks[:i])
        right = delimiter.join(chunks[i:])
        return [left, right]


def truncated_string(
    string: str,
    model: str,
    max_tokens: int,
    print_warning: bool = True,
):
    """Truncate a string to a maximum number of tokens."""
    encoding = tiktoken.encoding_for_model(model)
    encoded_string = encoding.encode(string)
    truncated_string = encoding.decode(encoded_string[:max_tokens])
    if print_warning and len(encoded_string) > max_tokens:
        print(f"Warning: Truncated string from {len(encoded_string)} tokens to {max_tokens} tokens.")
    return truncated_string


def split_strings_from_text(
    text: str,
    max_tokens: int = 1000,
    model: str = GPT_MODEL,
    max_recursion: int = 5,
):
    """
    Split a subsection into a list of subsections, each with no more than max_tokens.
    Each subsection is a tuple of parent titles [H1, H2, ...] and text (str).
    """
    string = text
    num_tokens_in_string = num_tokens(string)
    # if length is fine, return string
    if num_tokens_in_string <= max_tokens:
        return [string]
    # if recursion hasn't found a split after X iterations, just truncate
    elif max_recursion == 0:
        return [truncated_string(string, model=model, max_tokens=max_tokens)]
    # otherwise, split in half and recurse
    else:
        for delimiter in ["\n\n", "\n", ". "]:
            left, right = halved_by_delimiter(text, delimiter=delimiter)
            if left == "" or right == "":
                # if either half is empty, retry with a more fine-grained delimiter
                continue
            else:
                # recurse on each half
                results = []
                for half in [left, right]:
                    half_strings = split_strings_from_text(
                        half,
                        max_tokens=max_tokens,
                        model=model,
                        max_recursion=max_recursion - 1,
                    )
                    results.extend(half_strings)
                return results
    # otherwise no split was found, so just truncate (should be very rare)
    return [truncated_string(string, model=model, max_tokens=max_tokens)]

def get_document(id):
    variables = {
        'id': id,
    }
    query = """
        query GetDocument($id: ID!) {
            getDocument(id: $id) {
                id
                content
            }
        }
    """
    response = query_api(query, variables)
    return response['data']['getDocument']
