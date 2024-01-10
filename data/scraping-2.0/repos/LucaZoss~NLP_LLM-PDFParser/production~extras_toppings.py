# Sausage_999
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
from typing import Dict
from openai import OpenAI
import time
import json
import os
import spacy
from transformers import pipeline
import re

# Define your financial terms
terms = ['ISIN', 'Issuer', 'Currency', 'Underlying\(s\)', 'Strike', 'Launch Date',
         'Final Valuation Day', 'Maturity', 'Cap', 'Barrier', 'Bloomberg Code', 'ETI']

# Create regex patterns (case-insensitive)
patterns = {term: re.compile(term, re.IGNORECASE) for term in terms}


def extract_lines(text):
    lines = text.split('\n')
    matching_indices = set()
    for i, line in enumerate(lines):
        for pattern in patterns.values():
            if pattern.search(line):
                matching_indices.update(
                    {max(0, i-5), i, min(i+5, len(lines)-1)})
                break

    # Extract lines based on matching indices
    extracted_lines = [lines[i] for i in sorted(matching_indices)]
    return extracted_lines


# Example text (replace this with your actual text)
text = text

# Extract lines
sausage_999 = []
matching_lines = extract_lines(text)
for line in matching_lines:
    sausage_999.append(line)

sausage_999_string = str(" ".join(sausage_999))
print(type(sausage_999_string))
print(sausage_999_string)

# summarizers_transformers

# Assuming 'sausage_999_string' is your input string
input_text = sausage_999_string

# Split the text into chunks of approximately 1024 tokens
# This is a simplistic split and might need adjustment based on actual content
max_length = 1024
chunks = [input_text[i:i+max_length]
          for i in range(0, len(input_text), max_length)]

# Initialize the summarization pipeline
pipe = pipeline("summarization",
                model="nickmuchi/fb-bart-large-finetuned-trade-the-event-finance-summarizer")

# Summarize each chunk
summaries = [pipe(chunk)[0]['summary_text'] for chunk in chunks]

# Combine summaries (optional)
final_summary = ' '.join(summaries)

print(final_summary)

# Spacy Summarizer

# Load the Spacy model
nlp = spacy.load('en_core_web_sm')


# Process the text
doc = nlp(testing_sausage_1)
sausage_999 = []
# Extract entities
for ent in doc.ents:
    var = ent.text
    var_0 = ent.label_
    sausage_999.append(ent.text)
    # print(ent.text, ent.label_)
type(sausage_999)
print(sausage_999)


# Zero-Class-Classifier
classifier = pipeline("zero-shot-classification",
                      model="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli", use_fast=False)
candidate_labels = ['ISIN', 'Issuer', 'Ccy',
                    'Underlying(s)', 'Strike', 'Launch Date', 'Final Valuation Day', 'Maturity', 'Cap', 'Barrier']


# PLACE HOLDER


