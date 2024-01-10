import json
import openai
import os
import tiktoken



def process_text(user_input, context, maxtoken=100, project="flaskgeopolitics", location="us-central1"):
    # initialize

    # Start a chat sequence
    messages = [
        {"role": "system", "content": "You are an Information Analyst."},
        {"role": "system", "content": context},
        {"role": "user", "content": user_input}
    ]

    # Make a completion call
    oyyo = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=messages
    )

    # Print the assistant's response
    response = oyyo['choices'][0]['message']['content']


    # return response
    return response


def read_file(file_path):
    with open(file_path, "r") as file:
        return file.read()

def read_json(file_path):
    with open(file_path, "r") as file:
        return json.load(file)

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def trim_text_to_fit(string: str, max_tokens: int, encoding_name: str) -> str:
    """Trims the input string to fit within the specified number of tokens."""
    num_tokens = num_tokens_from_string(string, encoding_name)
    if num_tokens <= max_tokens:
        return string
    else:
        encoding = tiktoken.get_encoding(encoding_name)
        tokens = encoding.encode(string)
        return encoding.decode(tokens[:max_tokens])


max_tokens = 16000 # real max = 16348, we put some buffer here


data = read_json("/tmp/keepweb.txt")

for item in data:
    subject = item['subject']
    url = item['url']
    extracted_data = item['extracted_data']
    context_caption = """Please make headline caption not more than 3 words."""
    context_summarize = """Please make summarization over the input text."""

    if "browser supports JavaScript" in extracted_data or "JavaScript is disabled in this browser." in extracted_data or "404 Client Error" in extracted_data:
        # In this case, the page could not be scraped.
        caption = process_text(subject, context_caption)
        short_description = process_text(subject, context_summarize)
        print(f"{item['story_number']}: {caption}: {short_description}: {url}")
    else:
        # In this case, the page was successfully scraped.
        caption = process_text(subject, context_caption)
        trimmed_text = trim_text_to_fit(extracted_data, max_tokens, "cl100k_base")
        long_description = process_text(trimmed_text, context_summarize)
        print(f"{item['story_number']}: {caption}: {long_description}: {url}")

