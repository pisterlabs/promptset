import os
import sys
import time
import requests
import openai
from serpapi import GoogleSearch
from dotenv import load_dotenv
from termcolor import colored
from tqdm import tqdm
import colorama
colorama.init(autoreset=True)
from config import Config
import json
import re

load_dotenv()

cfg = Config()

# Configure OpenAI API key
try:
    openai.api_key = cfg.openai_api_key
    browserless_api_key = cfg.browserless_api_key
    llm_model = cfg.llm_model
except KeyError:
    sys.stderr.write("OpenAI key configuration failed.")
    exit(1)


def convert_to_gutenberg_blocks(text):
    # Split the text into lines
    lines = text.split('\n')

    # Initialize an empty list to hold the blocks
    blocks = []

    # Initialize an empty string to hold the current paragraph
    paragraph = ""

    # Iterate over each line
    for line in lines:
        # Check if the line is a headline
        if re.match(r"^# ", line):
            # If there's a current paragraph, add it as a block
            if paragraph:
                blocks.append(create_paragraph_block(paragraph))
                paragraph = ""

            # Add the headline as a block
            blocks.append(create_heading_block(line))

        # Check if the line is a list item
        elif re.match(r"^- ", line):
            # If there's a current paragraph, add it as a block
            if paragraph:
                blocks.append(create_paragraph_block(paragraph))
                paragraph = ""

            # Add the list as a block
            blocks.append(create_list_block(line))

        # Check if the line is an image
        elif re.match(r"^!\[.*\]\(.*\)", line):
            # If there's a current paragraph, add it as a block
            if paragraph:
                blocks.append(create_paragraph_block(paragraph))
                paragraph = ""

            # Add the image as a block
            blocks.append(create_image_block(line))

        # Check if the line is a quote
        elif re.match(r"^> ", line):
            # If there's a current paragraph, add it as a block
            if paragraph:
                blocks.append(create_paragraph_block(paragraph))
                paragraph = ""

            # Add the quote as a block
            blocks.append(create_quote_block(line))

        # Otherwise, add the line to the current paragraph
        else:
            paragraph += line

    # If there's a current paragraph, add it as a block
    if paragraph:
        blocks.append(create_paragraph_block(paragraph))

    # Return the blocks as a JSON string
    return json.dumps({"blocks": blocks}, ensure_ascii=False)

def create_paragraph_block(text):
    # Replace **text** with <strong>text</strong>
    text = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", text)
    return {
        "blockName": "core/paragraph",
        "attrs": {},
        "innerBlocks": [],
        "innerHTML": text
    }

def create_heading_block(text):
    return {
        "blockName": "core/heading",
        "attrs": {"level": text.count('#')},
        "innerBlocks": [],
        "innerHTML": text.replace('#', '').strip()
    }

def create_list_block(text):
    return {
        "blockName": "core/list",
        "attrs": {},
        "innerBlocks": [],
        "innerHTML": f"<li>{text.replace('-', '').strip()}</li>"
    }

def create_image_block(text):
    alt_text, url = re.match(r"^!\[(.*)\]\((.*)\)", text).groups()
    return {
        "blockName": "core/image",
        "attrs": {"url": url, "alt": alt_text},
        "innerBlocks": [],
        "innerHTML": ""
    }

def create_quote_block(text):
    return {
        "blockName": "core/quote",
        "attrs": {},
        "innerBlocks": [],
        "innerHTML": text.replace('> ', '').strip()
    }


def init(article, research):
    prompt = f"""The following article has been written by AI with user input. Review, if the user input is incorporated correctly.
    Rewrite the article if no. If yes, return NO CHANGE only.
    User input {research}\n\n
    Article: {article}"""

    chunked_output = ""
    for chunk in openai.ChatCompletion.create(
            model=cfg.llm_model,
            temperature=1,
            stream=True,
            messages=[
                {'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'system', 'content': "You're an expert in blogging, research and SEO."},
                {'role': 'system', 'content': 'Your name is BloggingGPT.'},
                {'role': 'system', 'content': 'Content you write is well SEO Optimised.'},
                {'role': 'system', 'content': 'You use engaging tone of voice.'},
                {"role": "user", "content": prompt}
            ]
    ):
        content = chunk["choices"][0].get("delta", {}).get("content")
        if content is not None:
            #print(content, end='')
            chunked_output += content
    return chunked_output

if __name__ == "__main__":
    # Open the file in read mode ('r')
    with open('your_file.txt', 'r') as file:
        # Read the file content and store it in the 'chunk' variable
        chunk = file.read()

    # Now you can use the 'chunk' variable in the convert_to_gutenberg_blocks function
    gutenberg_blocks = convert_to_gutenberg_blocks(chunk)