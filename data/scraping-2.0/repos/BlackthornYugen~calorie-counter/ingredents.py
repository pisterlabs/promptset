#!/usr/bin/env python3
import os
import sys
import argparse
import openai

parser = argparse.ArgumentParser(description="Generate two markdown tables with ingredients and densities.")
parser.add_argument("ingredients", nargs="?", type=argparse.FileType("r"), default=sys.stdin,
                    help="A list of ingredients. Reads from stdin if not specified.")
parser.add_argument("--model", default="gpt-4", type=str)
args = parser.parse_args()

openai.organization = "org-EkQHREB8jC9fCh451tq174pS"
openai.api_key = os.getenv("OPENAI_API_KEY")
print(openai.Model.list())
chat_model = openai.Model.retrieve(args.model)
print(f'\n\n\nMODEL:\n{chat_model}')

messages = [{
    "role": "system",
    "content": """When given a list of ingredients you create two markdown tables.
            First Table, titled "ingredients",  keep duplicates : item name, original unit/value,  weight in grams, density
            Second table, titled "Density", duplicates removed: item name, density"
            """
}, {
    "role": "user",
    "content": "".join(args.ingredients.readlines())
}]

completion = openai.ChatCompletion.create(
    model=chat_model.id,
    messages=messages
)
messages.append(
    {
        "role": completion['choices'][0]['message']['role'],
        "content": completion['choices'][0]['message']['content']
    }
)
print(f"\n\n\nCompletion:\n{completion}")
print(f"\n\n\nMessages:\n{messages}")
# print(f"{completion.choices[0].message.text}")

print(f"\n\n\n{completion['choices'][0]['message']['content']}")
