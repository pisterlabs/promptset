import openai
from concurrent.futures import ThreadPoolExecutor
import tiktoken
from dotenv import load_dotenv
import os
import json
import backoff  # for exponential backoff



load_dotenv()

os.environ['OPENAI_API'] = os.getenv('OPEN_AI_API')
# Add your own OpenAI API key
openai.api_key = os.environ['OPENAI_API']

def load_text(file_path):
    with open(file_path, 'r') as file:
        return file.read()

def save_to_file(responses, output_file):
    with open(output_file, 'w') as file:
        for response in responses:
            file.write(response + '\n')

# Change your OpenAI chat model accordingly

@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def call_openai_api(chunk):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=[
            {"role": "system", "content": "You are a smart technical writer who understands code and can write documentation for it."},
            {"role": "user", "content": f"Give me a developers documentation of the following code. Give a brief intro, table of contents, function explanations, dependencies, API specs (if present), schema tables in markdown. Give in markdown format and try to strict to the headings\n\n: {chunk}."},
        ],
        max_tokens=5000,
        n=1,
        stop=None,
        temperature=0.5,
    )
    return response.choices[0]['message']['content'].strip()

def split_into_chunks(text, tokens=500):
    encoding = tiktoken.encoding_for_model('gpt-3.5-turbo')
    words = encoding.encode(text)
    chunks = []
    for i in range(0, len(words), tokens):
        chunks.append(' '.join(encoding.decode(words[i:i + tokens])))
    return chunks

def process_chunks(text, output_file):
    # text = load_text(input_file)
    chunks = split_into_chunks(text)

    # Processes chunks in parallel
    with ThreadPoolExecutor() as executor:
        responses = list(executor.map(call_openai_api, chunks))

    save_to_file(responses, output_file)

# @backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def call_openai_api_higher_tokens(text, output_file):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=[
            {"role": "system", "content": "You are a smart technical writer who understands code and can write documentation for it."},
            {"role": "user", "content": f"Give me a developers documentation of the following code. Give a brief intro, table of contents, function explanations, dependencies, API specs (if present), schema tables in markdown. Give in markdown format and try to strict to the headings\n\n: {text}."},
        ],
        max_tokens=2000,
        n=1,
        stop=None,
        temperature=0.5,
    )
    save_to_file(response, output_file)
    # return response.choices[0]['message']['content'].strip()

# Specify your input text and output file path

# process_chunks(text, output_file)

# Can take up to a few minutes to run depending on the size of your data input