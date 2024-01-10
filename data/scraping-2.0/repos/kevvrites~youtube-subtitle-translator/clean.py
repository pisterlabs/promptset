import os
import openai
import json

openai.api_key = os.environ["OPENAI_API_KEY"]

source_language = "English"

with open('original-transcript.txt', 'r', encoding='utf-8') as transcript:
    original = transcript.readlines()

lines = [line.strip() for line in original if line.strip()]

system_prompt_template = "You will be given lines of text in {source}, and your task is to correct spelling errors, spacing errors. Preserve the original line formatting, specifically the number of lines in the file. The cleaned version should map to the original file as closely as possible."

system_prompt = system_prompt_template.format(source=source_language)

prompt = [{"role": "system", "content": system_prompt}]

for line in lines:
    prompt.append({"role": "user", "content": line})

response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=prompt,
    temperature=0,
)

with open('clean-transcript.txt', 'w+', encoding='utf-8') as f:
    f.write(response["choices"][0]["message"]["content"])
print('Cleaning transcript complete.\n')
