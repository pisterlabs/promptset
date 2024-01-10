#!/usr/bin/env python3
import argparse
import json
import logging
import textwrap
from pathlib import Path

import openai
from pylib import log
from tqdm import tqdm


def main():
    log.started()
    args = parse_args()

    with open(args.key_file) as f:
        keys = json.load(f)
    openai.api_key = keys["key"]

    labels = get_labels(args.text_dir)

    missed = 0
    for stem, text in tqdm(labels.items()):
        path = args.openai_dir / f"{stem}.json"
        if path.exists() and not args.overwrite:
            continue

        prompt = f'{args.prompt}, "{text}"'

        try:
            response = openai.ChatCompletion.create(
                model=args.model,
                messages=[
                    {"role": "system", "content": args.role},
                    {"role": "user", "content": prompt},
                ],
            )
            answer = response["choices"][0]["message"]["content"]
        except openai.error.Timeout:
            missed += 1
            continue

        with open(path, "w") as f:
            f.write(answer)

    logging.info(f"Missed {missed} labels")

    log.finished()


def get_labels(text_dir) -> dict[str, str]:
    labels = {}

    paths = sorted(text_dir.glob("*"))

    for path in paths:
        with open(path) as f:
            labels[path.stem] = f.read()

    return labels


def parse_args() -> argparse.Namespace:
    arg_parser = argparse.ArgumentParser(
        fromfile_prefix_chars="@",
        description=textwrap.dedent(
            """Use ChatGPT to extract trait information from museum label text."""
        ),
    )

    arg_parser.add_argument(
        "--text-dir",
        metavar="PATH",
        type=Path,
        required=True,
        help="""Directory containing the input text files.""",
    )

    arg_parser.add_argument(
        "--openai-dir",
        metavar="PATH",
        type=Path,
        required=True,
        help="""Output JSON files holding traits, one for each input text file, in this
            directory.""",
    )

    arg_parser.add_argument(
        "--key-file",
        metavar="PATH",
        type=Path,
        required=True,
        help="""This JSON file contains the OpenAI key.""",
    )

    arg_parser.add_argument(
        "--role",
        metavar="ROLE",
        default="You are an expert botanist.",
        help="""Prompt for ChatGPT. (default: %(default)s)""",
    )

    arg_parser.add_argument(
        "--prompt",
        metavar="PROMPT",
        default=(
            "Extract all information from the herbarium label text and put the output "
            "into JSON format using DarwinCore fields including dynamicProperties, "
        ),
        help="""Prompt for ChatGPT. (default: %(default)s)""",
    )

    arg_parser.add_argument(
        "--model",
        metavar="MODEL",
        default="gpt-4",
        help="""Which ChatGPT model to use. (default: %(default)s)""",
    )

    arg_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="""Overwrite any existing output JSON files.""",
    )

    args = arg_parser.parse_args()
    return args


if __name__ == "__main__":
    main()
