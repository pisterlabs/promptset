import os
import re
import argparse
import openai
from tqdm import tqdm
import sys


def chat_gpt(x, model, temperature, max_tokens, sys_message, api_key):
    openai.api_key = api_key
    response = openai.ChatCompletion.create(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": sys_message},
            {"role": "user", "content": x},
        ],
    )
    return response["choices"][0]["message"]["content"].strip()


def split_text(text, chunk_size):
    chunks = []
    lines = text.split("\n")
    word_count = 0
    chunk_start = 0

    for i, line in enumerate(lines):
        words_in_line = len(line.split())

        if word_count + words_in_line > chunk_size:
            chunk_end = i
            chunk = "\n".join(lines[chunk_start:chunk_end])

            # Find the nearest '---' sequence
            split_pos = chunk.rfind("---")

            # If not found, find the nearest blank line ''
            if split_pos == -1:
                split_pos = chunk.rfind("\n\n")

            # If not found, find the nearest end of a sentence
            if split_pos == -1:
                split_pos = max(
                    chunk.rfind("."), chunk.rfind("?"), chunk.rfind("!")
                )

            if split_pos != -1:
                chunk = chunk[: split_pos + 1]
                lines_in_chunk = len(chunk.split("\n"))
                word_count -= sum(
                    len(line.split())
                    for line in lines[
                        chunk_start : chunk_start + lines_in_chunk
                    ]
                )
                chunk_start = chunk_start + lines_in_chunk
            else:
                word_count = 0
                chunk_start = chunk_end

            chunks.append(chunk.strip())

        word_count += words_in_line

    # Add the remaining lines as the last chunk
    remaining_chunk = "\n".join(lines[chunk_start:])
    if remaining_chunk:
        chunks.append(remaining_chunk.strip())

    return chunks


def main():
    parser = argparse.ArgumentParser(
        description="Chunk a markdown file and run OpenAI completions on each chunk."
    )
    parser.add_argument("-i", "--input", help="Input markdown file.")
    parser.add_argument(
        "--content", help="Input markdown content via command line."
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=2000,
        help="Size of chunks to split the input file.",
    )
    parser.add_argument(
        "--model",
        default="gpt-3.5-turbo",
        help="OpenAI model to use for completions.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Temperature for OpenAI completions.",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=350,
        help="Max tokens for OpenAI completions.",
    )

    parser.add_argument(
        "--sys_message",
        default="You are Assistant who summarises any message concisely.",
        help="System message for OpenAI completions.",
    )
    parser.add_argument(
        "--api_key",
        default=os.getenv("OPENAI_API_KEY"),
        help="OpenAI API key.",
    )
    parser.add_argument(
        "-o", "--output", default=None, help="Output markdown file."
    )

    args = parser.parse_args()

    if args.input:
        with open(args.input, "r") as file:
            md_content = file.read()
    elif args.content:
        md_content = args.content
    else:
        md_content = sys.stdin.read()

    split_sections = split_text(md_content, args.chunk_size)

    summaries = []
    for section in tqdm(split_sections):
        try:
            summary = chat_gpt(
                section,
                args.model,
                args.temperature,
                args.max_tokens,
                args.sys_message,
                args.api_key,
            )
            summaries.append(summary)
        except Exception as e:
            print(e)

    output_text = "\n\n---\n\n".join(summaries)

    if args.output:
        with open(args.output, "w+") as outfile:
            outfile.write(output_text)
    else:
        print(output_text)


if __name__ == "__main__":
    main()
