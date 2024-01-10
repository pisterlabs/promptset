# %%
import re
from pdfminer.high_level import extract_text
from pathlib import Path
import random


def load_rules(rules_file=Path("data/raw/rules/MagicCompRules_21031101.pdf")):
    text = extract_text(rules_file)
    return text


def extract_rules(text: str) -> list[str]:
    see_rules_pattern = r"See rule \d+\.\d+\. |See rule \d+\.\d+"
    start_of_rule_pattern = r"\d+\.\d+\."

    processed_texts = re.sub(see_rules_pattern, "", text)
    rules = re.split(start_of_rule_pattern, processed_texts)
    # filter glossar and intro
    rules = rules[1:-23]
    rules = [rule.replace("\n", "") for rule in rules]

    print("random rule:")
    print(random.choice(rules))
    print("_________________")

    return rules


# %%

import numpy as np
import openai
import yaml

with open("config/config.yaml", "r") as infile:
    config = yaml.load(infile, Loader=yaml.FullLoader)

# roles: system, user, assistant
openai.api_key = config.get("open_ai_token")


def get_embeddings(rules: list[str]):
    text_embedding = []
    for rule in rules:
        response = openai.Embedding.create(input=rule, model="text-embedding-ada-002")
        embeddings = response["data"][0]["embedding"]
        text_embedding.append((rule, np.array(embeddings)))
    return text_embedding


# %%

text = load_rules()
rules = extract_rules(text)

# %%

text_embeddings = get_embeddings(rules[:2])

# %%

text_embeddings[0][1].shape

import hnswlib
