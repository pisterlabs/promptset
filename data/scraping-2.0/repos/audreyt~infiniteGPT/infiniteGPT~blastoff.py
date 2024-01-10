prompt = """
In the following text, remove unnecessary newlines.
Add two newslines between paragraphs.
Change the punctuation to use full-width ones such as ，。「」,
adding missing ones as necessary,
and keep the text as well as lines beginning with ###:

"""

prompt = """
In the following text, list all typos and inconsistencies as bullet points.

"""

import openai
from concurrent.futures import ThreadPoolExecutor
import tiktoken
import os
import random
import time

openai.api_key = os.environ.get("OPENAI_API_KEY")

def load_text(file_path):
    with open(file_path, 'r') as file:
        return file.read()

def save_to_file(responses, output_file):
    with open(output_file, 'w') as file:
        for response in responses:
            file.write(response + '\n')

def call_openai_api(chunk):
    while True:
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a world-class copywriter and proofreader, proficient in Traditional Mandarin and American English."},
                    {"role": "user", "content": f"{prompt}\n{chunk}"},
                ],
                max_tokens=2000,
                n=1,
                stop=None,
                temperature=0.1,
            )
            return response.choices[0].message.content.strip()
        except openai.error.RateLimitError as e:
            # Sleep for a random delay between 500ms and 1500ms before retrying
            delay = random.randint(500, 1500) / 1000
            print(f"Rate limit reached. Retrying in {delay:.2f}s...")
            time.sleep(delay)

def split_into_chunks(text, separator='### ', tokens=2000):
    encoding = tiktoken.encoding_for_model('gpt-4')
    paragraphs = text.split(separator)
    chunks = []
    for i, paragraph in enumerate(paragraphs):
        if i > 0:
            paragraph = f'{separator}{paragraph}'
        cnt = len(encoding.encode(paragraph))
        if cnt > tokens:
            subchunks = split_into_chunks(paragraph, '\n', tokens)
            chunks.extend(subchunks)
        elif len(chunks) == 0:
            chunks.append(paragraph)
        elif len(encoding.encode(f"{chunks[-1]}{paragraph}")) < tokens:
            chunks[-1] = f"{chunks[-1]}{paragraph}"
        else:
            chunks.append(paragraph)
    return chunks

def process_chunks(input_file, output_file):
    text = load_text(input_file)
    chunks = [chunk for chunk in split_into_chunks(text) if chunk.strip()]
    
    # Processes chunks in parallel
    with ThreadPoolExecutor() as executor:
        responses = list(executor.map(call_openai_api, chunks))

    save_to_file(responses, output_file)

# Specify your input and output files

if __name__ == "__main__":
    input_file = os.path.join(os.path.dirname(__file__), "test_input.txt")
    output_file = os.path.join(os.path.dirname(__file__), "output_og.txt")
    process_chunks(input_file, output_file)

# Can take up to a few minutes to run depending on the size of your data input
