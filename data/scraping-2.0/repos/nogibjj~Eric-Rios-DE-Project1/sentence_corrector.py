#!/usr/bin/env python

import openai
import os
import click

# Write a function that takes in a sentence and returns a corrected version of the sentence


def correct_sentence(sentence):
    """Corrects a sentence using OpenAI's GPT-3 API"""
    openai.api_key = os.getenv("OPENAI_API_KEY")
    response = openai.Completion.create(
        engine="davinci",
        prompt=f"""This is a sentence correction model. It will correct your sentence.

Sentence: {sentence}

Corrected sentence:""",
        temperature=0.5,
        max_tokens=100,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0.6,
        stop=["\n"],
    )
    return response.choices[0].text


@click.command()
@click.argument("sentence")
def main(sentence):
    """Corrects a sentence using OpenAI's GPT-3 API"""
    print(correct_sentence(sentence))


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()