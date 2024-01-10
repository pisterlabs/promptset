from pdfminer.high_level import extract_text
import os
import openai
import pandas as pd
import numpy as np
from openai_tasks.count_tokens import count_embedding_tokens
from dotenv import load_dotenv
import tiktoken
import spacy
load_dotenv()

nlp = spacy.load('en_core_web_sm')
api_key = os.getenv('OPENAI_API_KEY')
pdf = os.getenv('PDF_FILE_PATH')
pdf_text = os.getenv('PDF_TEXT')

# Function to extract text from PDF file


def extract_pdf_text(file_path):
    text = extract_text(file_path)
    with open('manifesto.txt', 'w') as f:
        f.write(text)
        return f


def sentence_df():
    extract_pdf_text(pdf)
    with open('manifesto.txt', 'r') as text:
        pdf_sentences = text.read()
        # Tokenize the text into sentences
        sentences = nlp(pdf_sentences)
        doc_sents = [sent.text for sent in sentences.sents]
        for doc_sent in doc_sents:
            doc_sent.replace("\n", " ")
        df = pd.DataFrame(data=doc_sents, columns=["sentences"])
        # Remove hyperlinks rows
        mask = df['sentences'].str.contains("hyperlinks")
        # Slice the DataFrame to exclude the rows that contain the word "Hyperlink"
        df = df[~mask]
        return df


def count_tokens(text):
    token_map = list()
    tokens = count_embedding_tokens(text)
    decoder = tiktoken.encoding_for_model(os.getenv("OPENAI_MODEL"))
    for token_list in tokens:
        token_map.append(decoder.decode(token_list))
    # Build a dataframe
    df = pd.DataFrame()
    df['decoded_tokens'] = token_map
    df['encoded_tokens'] = tokens
    return df


def manifesto_map_tokens_df():
    text = sentence_df()
    text = text['sentences'].tolist()
    count_tokens(text)


def openai_embedding_call(text):
    embeddings = list()
    openai.api_key = api_key
    for chunk in text[:-1]:
        embedding = openai.Embedding.create(
            input=chunk, model="text-embedding-ada-002"
        )["data"][0]["embedding"]
        embeddings.append(embedding)
    # For a DataFrame with 1536 (embeddings) columns and 132 (encodings) rows for ['encoded_tokens']
    embedding_array = np.array(embeddings)
    return embedding_array


def openai_pdf_embeddings():
    # Maximum tokens for the text-embedding-ada-002 model is 8191 per call
    text = manifesto_map_tokens_df()
    # Use encoded tokens
    text = text['encoded_tokens']
    openai_embedding_call(text)


# manifesto_embeddings_dataframe = openai_pdf_embeddings()

