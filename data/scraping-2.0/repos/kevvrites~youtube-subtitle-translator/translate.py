import os
import openai
import json

openai.api_key = os.environ["OPENAI_API_KEY"]

source_language = "English"
target_language = "French"

with open('clean-transcript.txt', 'r', encoding='utf-8') as file:
    transcript = file.read()

system_prompt_template = "You will be given lines of text in {source}, and your task is to translate them into {target}. Preserve the original line formatting, specifically the number of lines in the file. The translated version should map to the original file as closely as possible."

system_prompt = system_prompt_template.format(source=source_language, target=target_language)

prompt = [{"role": "system", "content": system_prompt}]

prompt.append({"role": "user", "content": transcript})

response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=prompt,
    temperature=0,
)

print(response)

with open('translated-transcript.txt', 'w+', encoding='utf-8') as f:
    f.write(response["choices"][0]["message"]["content"])

print('Translating transcript complete.\n')
