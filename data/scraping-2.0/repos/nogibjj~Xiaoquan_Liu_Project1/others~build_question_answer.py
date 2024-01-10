#!/usr/bin/env python

# shebang line to run the script every time in certain language
import os
import openai
import click


def submit_question(text):
    # write a function that uses open ar to generate a question and answer
    openai.api_key = "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"
    prompt = text

    response = openai.Completion.create(
        model="text-davinci-002",
        prompt=prompt,
        temperature=0.9,
        max_tokens=200,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=["\n", " Human:", " AI:"],
    )["choices"][0]["text"].strip(" \n")
    return response


@click.command()
@click.argument("text", type=str)
def main(text):
    """This is the main function that you ask the OPenAI API a question and to get an answer

    Example:  “What is the capital of France?”"""
    print(submit_question(text))


if __name__ == "__main__":
    main()
