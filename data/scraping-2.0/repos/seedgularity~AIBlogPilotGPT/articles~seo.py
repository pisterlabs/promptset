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

load_dotenv()

cfg = Config()

# Configure OpenAI API key
try:
    openai.api_key = cfg.openai_api_key
    browserless_api_key = cfg.browserless_api_key
    llm_model = cfg.llm_model
    fast_llm_model = cfg.fast_llm_model
except KeyError:
    sys.stderr.write("OpenAI key configuration failed.")
    exit(1)

def optimise_article(article):
    # Split the article into chunks of approximately 1000 words each
    article_chunks = [article[i:i+1000] for i in range(0, len(article), 1000)]

    optimised_article = ""

    for chunk in article_chunks:
        prompt = f"""You are an expert in SEO and blogging. A blogger wrote the following article.
        Your task is the review the article and optimise it for SEO. You don't return tips, you update the article by yourself.
        You can add, remove or change the text. Focus on long tail keywords and readability. 
        Article:{chunk}"""

        chunked_output = ""
        for response in openai.ChatCompletion.create(
                model=cfg.llm_model,
                temperature=0.9,
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
            content = response["choices"][0].get("delta", {}).get("content")
            if content is not None:
                chunked_output += content

        # Add the optimised chunk to the final article
        optimised_article += chunked_output

    return optimised_article