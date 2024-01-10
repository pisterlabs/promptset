"""Suggest ways to improve the article."""

import os
import sys
import openai

## Load the article ##
with open(sys.argv[-1], "r", encoding="utf-8") as f:
    article = f.read()

## Submit to ChatGPT ##
openai.api_key = os.getenv("OPENAI_SECRET")

completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {
            "role": "system",
            "content": (
                "You are an editor for a blog. Your job is to review my articles "
                "and offer me ways to improve my article. I want you to focus "
                "on making my articles more engaging and entertaining to read. "
                "Offer me at most 1-3 recommendations. In particular, offer me "
                "ways to reorder the content.\n\nTake the following article and "
                "offer ways of improving it. Do not respond with anything except "
                "the recommendations."
            ),
        },
        {
            "role": "user",
            "content": article,
        },
    ],
)

## Print the response ##
print(completion["choices"][0]["message"]["content"])
