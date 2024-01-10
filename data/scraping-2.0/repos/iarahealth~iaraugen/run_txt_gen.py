#!/usr/bin/env python3
import argparse
import random
import re

from openai import OpenAI
from typing import List
from tqdm import tqdm
from text.gen import make_chatgpt_query
from text.utils import post_process_sentences
from utils.files import append_sentences_to_file, read_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sentence/word generation using ChatGPT"
    )
    parser.add_argument(
        "--input_file", type=str, default=None, help="Input file with words"
    )
    parser.add_argument(
        "--num", type=int, default=None, help="Number of sentences or words to generate"
    )
    parser.add_argument(
        "--context",
        type=str,
        default="radiologia médica",
        help="Context of the generated sentences",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="A query to OpenAI's ChatGPT; the first number detected in the query will be replaced by the number of sentences to generate",
    )
    parser.add_argument(
        "--return_type",
        type=str,
        default="frases",
        help="Type of data to generate (default: frases)",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="OpenAI API key",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-3.5-turbo-16k",
        help="ChatGPT model to use",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=451,
        help="Random seed (default: 451)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file to write generated sentences",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    if args.query is None:
        if args.return_type == "frases":
            args.query = f"Você é um médico laudando. No contexto de {args.context}, gere {args.num} {args.return_type} contendo o termo '[MASK]', separadas por nova linha."
        else:
            args.query = f"No contexto de {args.context}, gere {args.num} {args.return_type} separadas por nova linha."
    else:
        args.num = (
            int(re.search(r"\d+", args.query).group())
            if re.search(r"\d+", args.query)
            else None
        )

    if args.input_file:
        wordlist = read_file(args.input_file)
    else:
        if args.return_type == "frases" and "[MASK]" in args.query:
            wordlist = []
            while True:
                word = input("Enter a word (or press Enter to finish): ")
                if word == "":
                    break
                wordlist.append(word)
        else:
            wordlist = [""]

    response_sentences: List[str] = []
    original_query = args.query
    openai_client = OpenAI(api_key=args.api_key)

    for word in tqdm(wordlist):
        word = word.strip()
        query = re.sub(r"\[MASK\]", word, original_query)
        number_of_sentences_left = args.num

        while number_of_sentences_left > 0:
            print(f"\nNumber of sentences left: {number_of_sentences_left}")
            print(f"Querying OpenAI's {args.model} with '{query}'...")
            query_response = make_chatgpt_query(
                openai_client,
                query,
                return_type=args.return_type,
                model=args.model,
            )
            print(query_response)
            response_sentences.extend(
                [s.split(" ", 1)[1] if s[0].isdigit() else s for s in query_response]
            )
            number_of_sentences_left -= len(query_response)
            query = re.sub(r"\d+", str(number_of_sentences_left), query)
        print()

    generated_sentences = post_process_sentences(response_sentences, modify=True)

    print("\nFinal results:")
    print("-------------------")
    for sentence in generated_sentences:
        print(sentence)
    print(f"\nTotal: {len(generated_sentences)} sentences")
    print("-------------------\n")

    if args.output:
        print(f"Appending generated sentences to {args.output}...")
        append_sentences_to_file(args.output, generated_sentences)
