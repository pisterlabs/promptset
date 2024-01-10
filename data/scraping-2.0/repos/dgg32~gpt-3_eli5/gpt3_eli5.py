import os
import openai
import sys

input_file = sys.argv[1]

openai.api_key = os.getenv("OPENAI_API_KEY")


def eli5 (text):
    header = "Summarize this for a second-grade student:\n\n"
    my_prompt = header + text

    response = openai.Completion.create(
    engine="text-davinci-002",
    prompt=my_prompt,
    temperature=0.7,
    max_tokens=150,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )

    return response.choices[0].text.lstrip()

with open(input_file, 'r') as file_in:
    for line in file_in:
        if len(line.strip()) > 0:
          res = eli5(line.strip() + "\n")

          print(res)