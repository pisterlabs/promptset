import glob
import json
import os
import pickle
import re

import openai
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from transformers import GPT2TokenizerFast
from og import get_og_data
from ask_embeddings import load_embeddings

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_TOKEN")

MODEL_NAME = "text-embedding-ada-002"
USELESS_TEXT_THRESHOLD = 100
SUBSTACK_URL = os.environ["SUBSTACK_URL"]


def strip_emoji(text: str):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)


def get_issue_slug(file_name):
    match = re.search(r"(?<=\.)[^.]*(?=\.)", file_name)
    if match:
        return match.group()
    return None


def get_embedding(text: str) -> list[float]:
    result = openai.Embedding.create(
        model=MODEL_NAME,
        input=text
    )
    return result["data"][0]["embedding"]


def compute_embedding(tokenizer, index, file):
    soup = BeautifulSoup(file, "html.parser")
    for sibling in soup.children:
        text = strip_emoji(sibling.get_text(" ", strip=True))
        if len(text) < USELESS_TEXT_THRESHOLD:
            continue
        embedding = get_embedding(text)
        return (text, embedding, len(tokenizer.tokenize(text)), index)


def get_issue_info(html_file):
    issue_slug = get_issue_slug(html_file)
    url = f"{SUBSTACK_URL}/p/{issue_slug}"
    og_data = get_og_data(url)
    return (url, og_data.get("og:image"), og_data.get("og:title"), og_data.get("og:description"))


def process_files():
    embeddings = []
    issue_info = {}
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    html_files = glob.glob("in/posts/*.html")

    for index, html_file in enumerate(html_files):
        print(f"Processing {html_file}...")
        with open(html_file, 'r') as file:
            issue_info[index] = get_issue_info(html_file)
            embeddings.append(compute_embedding(
                tokenizer, index, file))

    return embeddings, issue_info

def process_files_info_only():
    issue_info = {}

    html_files = glob.glob("in/posts/*.html")

    for index, html_file in enumerate(html_files):
        print(f"Processing {html_file}...")
        with open(html_file, 'r') as file:
            file_info = get_issue_info(html_file)
            print(json.dumps(file_info, indent=2))
            issue_info[index] = file_info

    return issue_info


def write_embeddings(embeddings, issue_info):
    with open('out/embeddings.pkl', 'wb') as f:
        pickle.dump({
            "embeddings": embeddings,
            "issue_info": issue_info
        }, f)

# embeddings = load_embeddings("in/embeddings.pkl")["embeddings"]
# issue_info = process_files_info_only()
embeddings, issue_info = process_files()
write_embeddings(embeddings, issue_info)
print("Done!")
