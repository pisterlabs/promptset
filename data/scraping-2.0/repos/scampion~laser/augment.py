import os
import json
import openai
from tqdm import tqdm
import joblib
from utils import get_pdf_files, get_text_from_pdf

openai.api_key = os.environ['OPENAI_API_KEY']

import openai

openai.api_type = "azure"
openai.api_base = "https://innovit-development.openai.azure.com/"
openai.api_version = "2023-07-01-preview"
openai.api_key = os.getenv("OPENAI_API_KEY")

memory = joblib.Memory('cache', verbose=0)

@memory.cache
def get_title(text, max_length=None):
    if max_length is None:
        max_length = len(text)
    message_text = [{"role":"system","content":"You are an AI assistant that helps people find information."},
                    {"role":"user","content": f"Give me a title for that content :\n\n\n {text[:max_length]}"}]
    completion = openai.ChatCompletion.create(
      engine="gpt-35-turbo",
      messages = message_text,
      temperature=0.7,
      max_tokens=800,
      top_p=0.95,
      frequency_penalty=0,
      presence_penalty=0,
      stop=None
    )
    return completion.choices[0]['message']['content']

@memory.cache
def get_summary(text, summary_length=300, max_length=None):
    if max_length is None:
        max_length = len(text)
    message_text = [{"role":"system","content":"You are an AI assistant that helps people find information."},
                    {"role":"user","content": f"Summarize that content in less than {summary_length} words:\n\n\n {text[:max_length]}"}]
    completion = openai.ChatCompletion.create(
      engine="gpt-35-turbo",
      messages = message_text,
      temperature=0.7,
      max_tokens=800,
      top_p=0.95,
      frequency_penalty=0,
      presence_penalty=0,
      stop=None
    )
    return completion.choices[0]['message']['content']


if __name__ == '__main__':
    data_dir = 'data'
    augmented_data = []
    for fp in tqdm(list(get_pdf_files(data_dir))):
        print("Process ", fp)
        url = 'https:/' + fp[len(data_dir):]
        text = get_text_from_pdf(fp)
        title = get_title(text, max_length=7000)
        print(title)
        summary = get_summary(text, max_length=7000)
        print(summary)
        augmented_data.append({"url": url,
                               "title": title.replace('Title: ', '').replace('"', ''),
                               "summary": summary})
    with open('augmented_data.json', 'w') as f:
        json.dump(augmented_data, f, indent=2)

