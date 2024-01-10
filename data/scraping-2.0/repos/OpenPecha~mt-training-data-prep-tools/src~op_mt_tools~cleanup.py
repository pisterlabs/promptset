import os
import re
import time
from collections.abc import Generator
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import List, Tuple

import backoff
import openai
import tiktoken

from op_mt_tools.tokenizers import en_sent_tokenizer

openai.api_key = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = "gpt-3.5-turbo-0301"
CONTEXT_LENGTH = 4096
CHUNK_MAX_TOKENS = 500  # smaller chunks give better result

# types
STR_WITH_SENT_PER_LINE = str


def num_tokens_from_messages(text, model=OPENAI_MODEL):
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")

    if model == "gpt-3.5-turbo":
        print(
            "Warning: gpt-3.5-turbo may change over time. Returning num tokens assuming gpt-3.5-turbo-0301."
        )
        return num_tokens_from_messages(text, model="gpt-3.5-turbo-0301")
    elif model == "gpt-4":
        print(
            "Warning: gpt-4 may change over time. Returning num tokens assuming gpt-4-0314."
        )
        return num_tokens_from_messages(text, model="gpt-4-0314")
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = (
            4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        )
    elif model == "gpt-4-0314":
        tokens_per_message = 3
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. \
                See https://github.com/openai/openai-python/blob/main/chatml.md \
                for information on how messages are converted to tokens."""
        )
    return len(encoding.encode(text)) + tokens_per_message


CLEANUP_PROMPT = """
Act as a text-cleaning pipeline. Strictly follow the cleaning steps below. Your input text is delimited by <>.

Cleaning Steps:
1 - remove extra spaces.
2 - remove only brackets and numbers.
3 - fix spelling errors.
4 - split text into sentences.
5 - join sentence which are split over multiple lines.

Output each sentence on a new line.
Do not report your steps and progress.

Input Text:
<{}>
"""


def split_document(document: str, chunk_max_tokens=CHUNK_MAX_TOKENS) -> List[str]:
    """Splits a document into chunks of text that are less than max_tokens long."""

    chunks = []
    current_chunk = []
    sents = en_sent_tokenizer(document).splitlines()
    for sentence in sents:
        current_chunk.append(sentence)
        # Check if the current chunk has more tokens than the limit
        tokens = num_tokens_from_messages(" ".join(current_chunk))
        if tokens > chunk_max_tokens:
            # If it exceeds the limit, remove the last added sentence and store the chunk
            current_chunk.pop()
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]

    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


@backoff.on_exception(backoff.expo, openai.OpenAIError)
def get_completion(prompt: str, model=OPENAI_MODEL) -> str:
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0,
        timeout=float("inf"),
    )
    return response.choices[0].message["content"]


def get_cleaned_sents(text: str, prompt_template=CLEANUP_PROMPT) -> List[str]:
    def parser_response(response: STR_WITH_SENT_PER_LINE) -> List[str]:
        return [sent.strip() for sent in response.splitlines() if sent]

    prompt = prompt_template.format(text).strip()
    response = get_completion(prompt)
    sents = parser_response(response)
    return sents


def get_chunks(
    text: str, chunks_dir: Path
) -> Generator[Tuple[int, int, str], None, None]:
    chunks_dir.mkdir(exist_ok=True)
    chunks_dir = chunks_dir.resolve()
    split_completed_marker = chunks_dir / "split_completed"
    if split_completed_marker.is_file():
        chunks_fns = sorted(chunks_dir.glob("*_chunk.txt"))
        for chunk_fn in chunks_fns:
            chunk_cleaned_file = chunk_fn.parent / f"{chunk_fn.stem}_cleaned.txt"
            if not chunk_cleaned_file.is_file():
                chunk_id = int(chunk_fn.stem.split("_")[0])
                yield chunk_id, len(chunks_fns), str(chunk_fn)
    else:
        chunks = split_document(text)
        for chunk_id, chunk in enumerate(chunks, start=1):
            chunk_fn = chunks_dir / f"{chunk_id:04}_chunk.txt"
            chunk_fn.write_text(chunk, encoding="utf-8")
            yield chunk_id, len(chunks), str(chunk_fn)
        split_completed_marker.touch()


def combine_chunks(chunks_dir: Path, output_fn: Path):
    text = ""
    for chunk_fn in sorted(chunks_dir.glob("*_chunk_cleaned.txt")):
        text += chunk_fn.read_text() + "\n"
    output_fn.write_text(text)


def cleanup_en_chunk(chunk: Tuple[int, int, str], chunks_dir: Path) -> None:
    chunk_id, chunks_len, chunk_fn = chunk
    start = time.time()
    print(f"\t- cleaning chunk {chunk_id}/{chunks_len} ...", end="")
    chunk_text = open(chunk_fn).read()
    sents = get_cleaned_sents(chunk_text)
    chunk_cleaned_fn = chunks_dir / f"{chunk_id:04}_chunk_cleaned.txt"
    chunk_cleaned_fn.write_text("\n".join(sents), encoding="utf-8")
    end = time.time()
    delta = end - start
    print(f" {delta:.3f}s")


def cleanup_en(
    fn: Path, cleaned_file_prefix: str = "[CLEANED]", verbose: bool = False
) -> Path:
    """Clean up english text using GPT-3."""
    chunks_dir = fn.parent / "chunks"
    cleaned_fn = fn.parent / f"{cleaned_file_prefix}_{fn.stem}.txt"
    text = fn.read_text(encoding="utf-8")
    doc_chunks = list(get_chunks(text, chunks_dir=chunks_dir))
    cleanup_en_chunk_dir_set = partial(cleanup_en_chunk, chunks_dir=chunks_dir)
    with ProcessPoolExecutor() as pool:
        _ = list(pool.map(cleanup_en_chunk_dir_set, doc_chunks))
    combine_chunks(chunks_dir, cleaned_fn)
    return cleaned_fn


def find_failed_cleanup_chunks(text_path: Path, overlap: float = 0.8) -> List[int]:
    """Find chunks that failed to clean up based on char count overlap

    Args:
        fn (Path): path to text_path
        overlap (float, optional): [description]. Defaults to 0.8.
    """

    def make_comparable(text: str) -> str:
        text = re.sub(r"\s{2,}", "", text)
        text = re.sub(r"\n", "", text)
        return text

    chunks_dir = text_path / "chunks"
    chunks_fns = sorted(chunks_dir.glob("*_chunk.txt"))
    failed_chunks = []
    for chunk_fn in chunks_fns:
        chunk_cleaned_fn = chunks_dir / f"{chunk_fn.stem}_cleaned.txt"
        if chunk_cleaned_fn.is_file():
            chunk = chunk_fn.read_text(encoding="utf-8")
            chunk_cleaned = chunk_cleaned_fn.read_text(encoding="utf-8")
            chunk_overlap = len(make_comparable(chunk_cleaned)) / len(
                make_comparable(chunk)
            )
            if chunk_overlap < overlap:
                chunk_id = int(chunk_fn.stem.split("_")[0])
                failed_chunks.append(chunk_id)

    return failed_chunks


def split_chunk_into_sentence(text_path: Path) -> None:
    chunks_dir = text_path / "chunks"
    chunks_fns = sorted(chunks_dir.glob("*_chunk.txt"))
    for chunk_fn in chunks_fns:
        text = chunk_fn.read_text(encoding="utf-8")
        sents_text = en_sent_tokenizer(text)
        chunk_sents_fn = chunks_dir / f"{chunk_fn.stem}_sents.txt"
        chunk_sents_fn.write_text(sents_text, encoding="utf-8")
