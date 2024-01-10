"""
Extracts clues from the question text given prompts and examples


Example usage:
python clue_extraction.py --prompt_text_file prompts/cover.prompt.txt --examples_file prompts/examples/cover_clue_examples.jsonl  --input_file ./data/2022-05-30_14441_gold_posts.cover_clues.jsonl --output_file ./data/debug.2022-05-30_14441_gold_posts.cover_clues.jsonl  --max_examples 1
"""
import argparse
import os
from typing import Callable, Dict, List, Union

import backoff
import jsonlines
import openai
from tqdm import tqdm

openai.api_key = os.getenv("OPENAI_API_KEY")


def date_examples_to_string(examples: List[Dict[str, str]]) -> str:
    return "\n".join(
        [
            f"Question: {example['question']} Question Posted Date: {example['question_posted_date']} Date Clue: {example['date_clue']}"
            for example in examples
        ]
    )


def test_date_examples_to_string(example: Dict[str, str]) -> str:
    return f"\nQuestion: {example['question']} Question Posted Date: {example['question_posted_date']} Date Clue:"


def plot_examples_to_string(examples: List[Dict[str, str]]) -> str:
    return "\n".join(
        [
            f"Question: {example['question']} Plot Clue: {example['plot_clue']}"
            for example in examples
        ]
    )


def test_plot_examples_to_string(example: Dict[str, str]) -> str:
    return f"\nQuestion: {example['question']} Plot Clue:"


def date_extract_examples_to_string(examples: List[Dict[str, str]]) -> str:
    return "\n".join(
        [
            f"Question: {example['question']} Date Clue: {example['date_clue']}"
            for example in examples
        ]
    )


def test_date_extract_examples_to_string(example: Dict[str, str]) -> str:
    return f"\nQuestion: {example['question']} Date Clue:"


def cover_examples_to_string(examples: List[Dict[str, str]]) -> str:
    return "\n".join(
        [
            f"Question: {example['question']} Cover Clue: {example['cover_clue']}"
            for example in examples
        ]
    )


def test_cover_examples_to_string(example: Dict[str, str]) -> str:
    return f"\nQuestion: {example['question']} Cover Clue:"


def title_examples_to_string(examples: List[Dict[str, str]]) -> str:
    return "\n".join(
        [
            f"Question: {example['question']} Title Clue: {example['title_clue']}"
            for example in examples
        ]
    )


def test_title_examples_to_string(example: Dict[str, str]) -> str:
    return f"\nQuestion: {example['question']} Title Clue:"


def author_examples_to_string(examples: List[Dict[str, str]]) -> str:
    return "\n".join(
        [
            f"Question: {example['question']} Author Clue: {example['author_clue']}"
            for example in examples
        ]
    )


def test_author_examples_to_string(example: Dict[str, str]) -> str:
    return f"\nQuestion: {example['question']} Author Clue:"


def genre_examples_to_string(examples: List[Dict[str, str]]) -> str:
    return "\n".join(
        [
            f"Question: {example['question']} Genre Clue: {example['genre_clue']}"
            for example in examples
        ]
    )


def test_genre_examples_to_string(example: Dict[str, str]) -> str:
    return f"\nQuestion: {example['question']} Genre Clue:"


examples_to_string: Dict[str, Callable[[List[Dict[str, str]]], str]] = {
    "date": date_examples_to_string,
    "cover": cover_examples_to_string,
    "title": title_examples_to_string,
    "date_extract": date_extract_examples_to_string,
    "plot_extract": plot_examples_to_string,
    "author": author_examples_to_string,
    "genre": genre_examples_to_string,
}

test_example_to_string: Dict[str, Callable[[Dict[str, str]], str]] = {
    "date": test_date_examples_to_string,
    "cover": test_cover_examples_to_string,
    "title": test_title_examples_to_string,
    "date_extract": test_date_extract_examples_to_string,
    "plot_extract": test_plot_examples_to_string,
    "author": test_author_examples_to_string,
    "genre": test_genre_examples_to_string,
}


@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def get_clue(
    example: Dict[str, str],
    examples: List[Dict[str, str]],
    prompt_pre: str,
    prompt_post: str,
    metadata_field: str,
    max_tokens: int = 10000,
) -> str:
    prompt = prompt_pre
    prompt += examples_to_string[metadata_field](examples)
    prompt += "\n\n" + prompt_post
    prompt += test_example_to_string[metadata_field](example)

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k-0613",
        max_tokens=max_tokens,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )

    return response["choices"][0]["message"]["content"]


def extract_clues_from_file(
    prompt_text_file: str,
    examples_file: str,
    input_file: str,
    output_file: str,
    metadata_field: str,
    max_examples: Union[None, int] = None,
    max_tokens: int = 10000,
) -> None:
    with open(prompt_text_file, "r") as f:
        prompts = f.readlines()

    with jsonlines.open(examples_file) as reader:
        examples = [obj for obj in reader]

    with jsonlines.open(input_file) as reader:
        with jsonlines.open(output_file, "w") as writer:
            for idx, obj in tqdm(enumerate(reader)):
                if max_examples is not None and idx >= max_examples:
                    exit()
                else:
                    try:
                        clue = get_clue(
                            example={
                                "question": obj["comments"][0]["comment_text"],
                                "question_posted_date": obj["comments"][0][
                                    "parsed_date"
                                ],
                            },
                            prompt_pre=prompts[0],
                            prompt_post=prompts[1],
                            examples=examples,
                            metadata_field=metadata_field,
                            max_tokens=max_tokens,
                        )
                        obj["comments"] = [{"comment_text": clue}]
                        writer.write(obj)
                    except Exception as e:
                        print(e)
                        print(obj)
                        continue


## make main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract clues from question text")
    parser.add_argument(
        "--prompt_text_file",
        type=str,
        help="The first line will be before the examples and the second line will be after the examples",
    )
    parser.add_argument("--examples_file", type=str, help="examples file path")
    parser.add_argument("--input_file", type=str, help="input file path")
    parser.add_argument("--output_file", type=str, help="output file path")
    parser.add_argument("--metadata_field", type=str, help="data, image, etc.")
    parser.add_argument(
        "--max_examples", type=int, help="max number of examples to use"
    )
    parser.add_argument("--max_tokens", type=int, help="max tokens to generate")

    args = parser.parse_args()
    extract_clues_from_file(
        args.prompt_text_file,
        args.examples_file,
        args.input_file,
        args.output_file,
        args.metadata_field,
        args.max_examples,
        args.max_tokens,
    )
    extract_clues_from_file(
        prompt_text_file="prompts/date.prompt.txt",
        examples_file="prompts/examples/date_clue_examples.jsonl",
        input_file="data/2022-05-30_14441_gold_posts.jsonl",
        output_file="...",
        metadata_field="date",
        max_examples=1,
    )
