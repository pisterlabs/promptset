# Auto-Archive: Document Processor Module
# Path: `\core\document_processor.py`
# Version: `v0.0.3-dev`
# Updated: 08-08-2023

# 🚧 ACTIVE DEVELOPMENT 🚧 #

## ## ## ## ## ## ## ## ## ## ## ## ##

import openai
import os
from tokenizers import Tokenizer, models, pre_tokenizers, decoders
from docx import Document
from datetime import datetime
import json
from scipy.spatial.distance import cosine
import spacy
import logging

# Import log configuration
from core.log_config import document_processor_logger

# Initialize spaCy model
nlp = spacy.load("en_core_web_sm")

# OpenAI API key
openai.api_key = os.environ['OPENAI_API_KEY']

# Define the categories and their representative prompts
CATEGORIES_PROMPTS = {
    1: "This is a personal narrative or memoir, recounting the life experiences of the author.",
    2: "This is a detailed account of a person's life, written by someone else.",
    3: "This is a personal record of events, experiences, thoughts, and observations.",
    4: "This contains legal information or documents, such as contracts, legal opinions, laws, or judicial decisions.",
    5: "This is a written communication between individuals, typically containing personal or professional information.",
    6: "This is a detailed record of the events and discussions at a formal meeting.",
    7: "This is a publication, usually issued daily or weekly, containing news and current events.",
}

# Define the threshold for assigning a category
CATEGORY_THRESHOLD = 0.5

# Initialize the tokenizer
tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
tokenizer.decoder = decoders.ByteLevel()

# Function to count tokens using HuggingFace tokenizer


def count_tokens(text):
    output = tokenizer.encode(text)
    print("Counting tokens...")
    return len(output.tokens)


# The maximum number of tokens that a model can handle
MAX_TOKENS = 10000

# Function to split the text into chunks


def split_into_chunks(text, max_tokens):
    if not isinstance(text, str):
        raise TypeError(f"Expected string but received {type(text)}")
    tokens = tokenizer.encode(text).tokens
    token_list = list(tokens)

    chunks = []
    current_chunk = []

    for token in token_list:
        if len(current_chunk) + len(token) <= max_tokens:
            current_chunk.append(token)
        else:
            chunks.append(current_chunk)
            current_chunk = [token]

    chunks.append(current_chunk)
    chunks = [''.join(chunk) for chunk in chunks]

    print("Segmenting text...")
    return chunks

# Function to generate embeddings


def generate_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("", " ")
    return openai.Embedding.create(input=[text], model=model)['data'][0]['embedding']

# Function to calculate cosine similarity between two embeddings


def calculate_similarity(embedding1, embedding2):
    return 1 - cosine(embedding1, embedding2)

# Function to extract entities using spaCy


def extract_entities(text):
    logging.info("Extracting entities...")
    doc = nlp(text)
    entities = [{'entity': ent.text, 'type': ent.label_} for ent in doc.ents]
    logging.info(f"Entities extracted: {entities}")
    return entities

# Chat-based approach for document processing


def process_document(filepath):
    _, file_extension = os.path.splitext(filepath)
    print("Processing document...")

    if file_extension == '.docx':
        print("File type: Word (.docx) document.")
        document = Document(filepath)
        document_text = " ".join(
            paragraph.text for paragraph in document.paragraphs)
    elif file_extension == '.txt':
        print("File type: Text (.txt) document.")
        with open(filepath, 'r', encoding='utf-8') as file:  # use 'utf-8' encoding
            document_text = file.read()
    else:
        print("File type: Unsupported. Error.")
        raise Exception(f"Unsupported file type: {file_extension}")

    summary = generate_summary(document_text)
    summary_title = generate_summary_title(summary)
    summary_embedding = generate_embedding(summary)
    category_id = classify_document(summary_embedding)
    entities = extract_entities(document_text)
    archive_profile = create_archive_profile(
        document_text, filepath, summary, summary_title, category_id, summary_embedding)

    return document_text, archive_profile, summary_embedding, entities

# Chat-based approach for generating summary


def generate_summary(document_text):
    print("Generating document summary...")

    if not isinstance(document_text, str):
        raise TypeError(f"Expected string but received {type(document_text)}")

    # Check the number of tokens in the document text
    num_tokens = count_tokens(document_text)
    ("Analyzing document length...")
    if num_tokens > MAX_TOKENS:
        print("Segmenting document...")
        chunks = split_into_chunks(document_text, MAX_TOKENS)
        print("Document segmented.")
    else:
        chunks = [document_text]

    summaries = []

    # Generate a summary for each chunk
    for chunk in chunks:
        print("Analying document segments...")
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Please provide a detailed summary of the following text: {chunk}"}
        ]
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k",
            messages=messages,
            max_tokens=MAX_TOKENS,
            n=1,
            temperature=0.2
        )
        summary = response['choices'][0]['message']['content']
        summaries.append(summary)

    # Prepare a list of messages for the master summary
    master_summary_messages = [
        {"role": "system", "content": "You are a helpful assistant."}]
    for summary in summaries:
        master_summary_messages.append(
            {"role": "user", "content": f"Write a summary of the following: {summary}"})

    master_summary_response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=master_summary_messages,
        max_tokens=1000,
        n=1,
        temperature=0.2
    )
    full_summary = master_summary_response['choices'][0]['message']['content']

    print("Summary generated.")
    return full_summary

# Chat-based approach for generating summary title


def generate_summary_title(summary):
    print("Generating document title...")

    # Check the number of tokens in the summary
    num_tokens = count_tokens(summary)
    if num_tokens > MAX_TOKENS:
        print(
            f"Warning: Generated summary contains {num_tokens} tokens, which exceeds the model's maximum of {MAX_TOKENS}. Truncating summary.")
        summary = tokenizer.decode(tokenizer.encode(summary)[:MAX_TOKENS])

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"Generate a title for the document with the following summary: {summary}"}
    ]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=messages,
        max_tokens=MAX_TOKENS,
        n=1,
        temperature=0.3
    )

    summary_title = response['choices'][0]['message']['content']
    print("Title generated.")
    return summary_title

# Chat-based approach for classifying document


def classify_document(embedding):
    print("Classifying document...")
    scores = {}
    for category_id, prompt in CATEGORIES_PROMPTS.items():
        # Catch exceptions when the API call fails
        try:
            prompt_embedding = generate_embedding(prompt)
            score = calculate_similarity(embedding, prompt_embedding)
            scores[category_id] = score
        except Exception as e:
            print(f"Error: {e}")
            scores[category_id] = 0

    # Assign the category with the highest score to the document
    # If none of the categories score above the threshold, assign the "Other" category
    sorted_scores = sorted(
        scores.items(), key=lambda item: item[1], reverse=True)
    if sorted_scores[0][1] >= CATEGORY_THRESHOLD:
        category_id = sorted_scores[0][0]
    else:
        category_id = 8  # "Other"

    print("Document classified.")
    return category_id

# Function to create an archive profile


def create_archive_profile(document_text, filename, summary, summary_title, category_id, embedding):
    print("Creating archive profile...")
    profile_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    archive_profile = {
        'profile_title': summary_title,
        'profile_summary': summary,
        'profile_text': document_text,
        'profile_date': profile_date,
        'category_id': category_id,
        # embedding is already a list
        'profile_embedding': json.dumps(embedding)
    }

    print("Archive profile created.")
    print("Document processed successfully.")
    return archive_profile
