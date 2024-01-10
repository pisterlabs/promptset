#!/usr/bin/env python3

"""
Code for the MAS.S68 (Generative AI for Constructive Communication) programming workshop

Reverse dictionary (description-to-word guesser) using the OpenAI GPT-3 API

Prompts and evaluation data are in data/train.jsonl and data/test.jsonl respectively
and were produced by ./pull_rd_data.py
"""

import argparse
import json
import os
import re

import openai
import streamlit as st

# Don't forget to set your OPENAI_API_KEY environment variable.
# Or set it here directly (but don't check it into a git repo.)
openai.api_key = os.getenv("OPENAI_API_KEY")


def call_gpt3_api(prompt, model):
    response = openai.Completion.create(
        model=model,
        prompt=prompt,
        temperature=0,
        max_tokens=256,
        frequency_penalty=0,
        presence_penalty=0,
    )
    return response


def definition_to_zero_shot_prompt(definition):
    return 'What are some words that mean "%s"?\n\n' % (definition)


def definition_to_few_shot_prompt(definition, examples):
    instructions = "Please find words that mean the same thing as the given definition.  For example:\n\n"
    target = {
        "definition": definition,
        "word": "",
    }  # placeholder for the word we're looking for
    # Format the example text as a bulleted list
    example_text = "\n".join(
        [
            '- "%s": %s' % (example["definition"], example["word"])
            for example in examples + [target]
        ]
    )
    return instructions + example_text


def response_to_completion_text(openai_response):
    return openai_response["choices"][0]["text"]


def completion_text_to_words(s):
    words = re.sub("[^A-Za-z]", " ", s.strip()).split()
    if words:
        return words
    else:
        # Return empty string if it produces no answer
        return []


def get_words_for_definition(definition, examples_to_use=None):
    if not examples_to_use:
        prompt = definition_to_zero_shot_prompt(definition)
    else:
        prompt = definition_to_few_shot_prompt(definition, examples_to_use)

    openai_response = call_gpt3_api(prompt, args.model)
    completion = response_to_completion_text(openai_response)
    return completion_text_to_words(completion)


def read_batch_of_queries(filename):
    """Reads a set of reverse dictionary labeled data as a list of dictionaries."""
    return [json.loads(line) for line in open(filename, "r").read().strip().split("\n")]


def get_example_queries_for_prompt(filename):
    return read_batch_of_queries(filename)[: args.num_prompt_examples]


def run_batch_of_queries(evaluation_queries_filename, prompt_example_queries_filename):
    evaluation_queries = read_batch_of_queries(evaluation_queries_filename)
    if args.num_prompt_examples > 0:
        prompt_example_queries = get_example_queries_for_prompt(
            prompt_example_queries_filename
        )
    else:
        prompt_example_queries = None
    num_correct = 0
    for record in evaluation_queries:
        definition = record["definition"]
        record["gpt3_words"] = get_words_for_definition(
            definition, examples_to_use=prompt_example_queries
        )
        record["gpt3_is_correct"] = (
            len(record["gpt3_words"]) > 0
            and record["gpt3_words"][0].lower() == record["word"].lower()
        )
        if record["gpt3_is_correct"]:
            num_correct += 1
        print(json.dumps(record))
    print(
        "Accuracy = %d / %d = %f"
        % (num_correct, len(evaluation_queries), num_correct / len(evaluation_queries))
    )


def streamlit_app():
    with st.form("main_form"):
        query = st.text_input("Enter description of the word you're looking for")
        submitted = st.form_submit_button("Submit")
        if submitted:
            st.write(get_words_for_definition(query))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--query", help="Look up words matching this description", default=None
    )
    parser.add_argument(
        "--eval",
        help="Run an evaluation on the given file of queries (in jsonl format)",
        default=None,
    )
    parser.add_argument(
        "--model", help="Which OpenAI model to use", default="text-curie-001"
    )
    parser.add_argument(
        "--num_prompt_examples",
        help="The number of examples from data/train.jsonl to include in the prompt.  If 0, use a separate 0-shot prompt.",
        default=0,
    )
    args = parser.parse_args()

    args.num_prompt_examples = int(args.num_prompt_examples)
    if args.query:
        print(
            get_words_for_definition(
                args.query, get_example_queries_for_prompt("data/train.jsonl")
            )
        )
    elif args.eval:
        run_batch_of_queries(args.eval, "data/train.jsonl")
    else:
        streamlit_app()
