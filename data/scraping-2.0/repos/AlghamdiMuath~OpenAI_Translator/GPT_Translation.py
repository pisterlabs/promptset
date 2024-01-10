#!/usr/bin/env python
# coding: utf-8

# Importing necessary libraries
import textwrap
import os
import openai
import tiktoken
from dotenv import load_dotenv, find_dotenv

# Load environment variables from a .env file
_ = load_dotenv(find_dotenv())

# Function to wrap text to a specified width
def wrap_text_to_fixed_width(text, width=120):
    wrapper = textwrap.TextWrapper(width=width)
    return wrapper.fill(text=text)

# Function to tokenize the content of a given text file
def tokenize_text_from_file(file_path, model):
    with open(file_path, 'r', encoding="utf8") as file:
        text = file.read()
    encoding = tiktoken.get_encoding(model)
    tokens = encoding.encode(text)
    return tokens

# Function to split tokens into smaller chunks based on a given size
def partition_tokens_into_chunks(tokens, max_chunk_size):
    num_chunks = (len(tokens) + max_chunk_size - 1) // max_chunk_size
    chunks = [tokens[i * max_chunk_size:(i + 1) * max_chunk_size] for i in range(num_chunks)]
    return chunks

# Function to convert token chunks back into text form
def convert_chunks_to_text(token_chunks, model):
    encoding = tiktoken.get_encoding(model)
    text_chunks = [encoding.decode(chunk) for chunk in token_chunks]
    return text_chunks

# Function to get translated text using OpenAI's model
def get_translated_text(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    return response.choices[0].message["content"]

# Function to execute the entire translation process
def execute_translation(template, text_to_translate):
    query = template + text_to_translate
    arabic_text = get_translated_text(query)
    return arabic_text

# Main Execution Code
if __name__ == "__main__":
    input_file_path = 'INPUT_FILE'
    output_file_path = 'OUTPUT_FILE'
    max_chunk_size = 1000
    model_name = "cl100k_base"

    # Tokenization and Chunking
    tokens = tokenize_text_from_file(input_file_path, model_name)
    token_chunks = partition_tokens_into_chunks(tokens, max_chunk_size)
    text_chunks = convert_chunks_to_text(token_chunks, model_name)

    translation_template = '''You are a professional translator,
    you excel in translating from English to Arabic
    word for word maintaining the structure and the context.\
    Your task is to translate the following English text to \
    Arabic perfectly and without missing any words.\

    English text: '''
    
    # Translation and Writing to File
    for text_chunk in text_chunks:
        translated_text = execute_translation(translation_template, text_chunk)
        with open(output_file_path, 'a', encoding='utf-8') as output_file:
            output_file.write(translated_text)
