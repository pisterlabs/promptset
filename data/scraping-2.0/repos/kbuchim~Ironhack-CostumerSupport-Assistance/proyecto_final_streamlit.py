import streamlit as st
import os
import openai
from dotenv import load_dotenv

import pandas as pd
from typing import Set
from transformers import GPT2TokenizerFast

import argparse, sys
import numpy as np

import PyPDF2
from PyPDF2 import PdfReader
import csv
import pickle

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import requests
from bs4 import BeautifulSoup

import sys
import nltk
from pdfminer.high_level import extract_text, extract_pages
from pdfminer.layout import LTTextContainer
from pdfminer.high_level import extract_pages
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from io import StringIO

import nltk
nltk.download('punkt')

from nltk.tokenize.punkt import PunktSentenceTokenizer
tokenizer = PunktSentenceTokenizer()

# Use load_env to trace the path of .env:
load_dotenv('.env')

openai.organization = "org-BJVQfnJYTuAJz2TkTNbchIf2"
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.Model.list()

def parser(file_path):
        # Parse the PDF file
    reader = PyPDF2.PdfReader(file_path)
    # Loop over each page in the PDF document
    sentences = []
    for page in range(len(reader.pages)):
        # Extract the text from the page
        pdf_text = reader.pages[page].extract_text()
        
        # in my case, when it had '\n', I called it a new paragraph, 
        # like a collection of sentences
        paragraphs = [p for p in pdf_text.split('\n') if p]
        # and here, sent_tokenize each one of the paragraphs
        for paragraph in paragraphs:
            pdf_sentences = tokenizer.tokenize(paragraph)
    
            # Add the sentences to the list of sentences
            sentences.extend(pdf_sentences)
    return sentences

def search_question(sentences, question):

    # Search for the question in the sentences
    best_sentence = None
    best_score = 0
    for sentence in sentences:
        # Calculate the score for the sentence based on the number of overlapping words with the question
        score = len(set(re.findall(r'\b\w+\b', sentence.lower())) & set(re.findall(r'\b\w+\b', question.lower())))
        # Update the best sentence and score if this sentence has a higher score than the current best sentence
        if score > best_score:
            best_sentence = sentence
            best_score = score
    # Return the best sentence as the answer
    return best_sentence

def get_text_lines(pdf_file):
    """
    Obtiene todas las líneas de texto horizontales de un PDF
    """
    text_lines = []
    for page_layout in pdf_layout:
        for element in page_layout:
            if isinstance(element, LTTextBoxHorizontal):
                for text_line in element:
                    if isinstance(text_line, LTTextLineHorizontal):
                        text_lines.append(text_line)
    return text_lines

def unify_text_lines(text_lines):
    """
    Unifica varias líneas de texto en una sola frase
    """
    # Ordena las líneas de texto por su posición vertical
    text_lines.sort(key=lambda x: -x.y0)
    
    # Concatena el contenido de las líneas de texto
    text = ' '.join(line.get_text().strip() for line in text_lines)
    print(text)
    return text


tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

def count_tokens(text: str) -> int:
    """count the number of tokens in a string"""
    return len(tokenizer.encode(text))

def extract_page(page: str,
    index: int
) -> str:
    
    """
    Extracts the content and token count from the given page
    """
    content = ' '.join([el.get_text().strip() for el in page if isinstance(el, LTTextContainer)])
    token_count = count_tokens(content) + 4 # adding 4 extra tokens
    return ("Page " + str(index), content, token_count)


COMPLETIONS_MODEL = "text-davinci-003"

MODEL_NAME = "curie"

DOC_EMBEDDINGS_MODEL = f"text-search-{MODEL_NAME}-doc-001"
QUERY_EMBEDDINGS_MODEL = f"text-search-{MODEL_NAME}-query-001"

def get_embedding(text: str, model: str=DOC_EMBEDDINGS_MODEL) -> list[float]:
    result = openai.Embedding.create(
      model=model,
      input=text
    )
    return result["data"][0]["embedding"]

def compute_doc_embeddings(df: pd.DataFrame) -> dict[tuple[str], list[float]]:
    """
    Create an embedding for each row in the dataframe using the OpenAI Embeddings API.
    Return a dictionary that maps between each embedding vector and the index of the row that it corresponds to.
    """
    embeddings_dict = {}
    for idx, r in df.iterrows():
        content = r["content"]
        embedding = get_embedding(content)
        embeddings_dict[(content, idx)] = embedding
    return embeddings_dict
    return {
        idx: get_embedding(r.content) for idx, r in df.iterrows()
    }




from sklearn.metrics.pairwise import cosine_similarity

MAX_SECTION_LEN = 500
SEPARATOR = "\n* "
separator_len = 3

def construct_prompt(question: str, context_embeddings: dict, df: pd.DataFrame, section_index) -> tuple[str, str]:
    document_section = df.iloc[section_index]
    chosen_sections = []
    chosen_sections_len = 0
    chosen_sections_indexes = []

    chosen_sections_len += document_section.tokens + separator_len
    if chosen_sections_len > MAX_SECTION_LEN:
        space_left = MAX_SECTION_LEN - chosen_sections_len - len(SEPARATOR)
        chosen_sections.append(SEPARATOR + document_section.content[:space_left])
        chosen_sections_indexes.append(str(section_index))

    chosen_sections.append(SEPARATOR + document_section.content)
    chosen_sections_indexes.append(str(section_index))
    
    header = 'Manten tus respuestas en máximo 3 oraciones. Se conciso, y completa siempre las oraciones. \n Este es un contexto que puede ser útil :\n'
    
    return (header + "".join(chosen_sections) + "\n\n\nQ: " + question + "\n\nA: "), ("".join(chosen_sections))


    

COMPLETIONS_API_PARAMS = {
    # We use temperature of 0.0 because it gives the most predictable, factual answer.
    "temperature": 0.0,
    "max_tokens": 150,
    "model": COMPLETIONS_MODEL,
} 

def answer_query_with_context(query, df, embeddings):
      # Compute query embedding
    #query_embedding = np.mean(embeddings.embed_sentences([query]), axis=0)
    query_embedding = np.array(get_embedding(query))
    

    # Compute cosine similarity between query embedding and all document embeddings
    #similarities = cosine_similarity(embeddings.embedding_matrix, query_embedding.reshape(1, -1))
    similarities = cosine_similarity(list(embeddings.values()), query_embedding.reshape(1,-1))
    
    # Find index of most similar document
    most_similar_index = np.argmax(similarities)
    
    print(most_similar_index)
    
    #Construct Prompt
    prompt, context = construct_prompt(
        query,
        embeddings,
        df,
        most_similar_index
    )

    print("===\n", prompt)

    response = openai.Completion.create(
                prompt=prompt,
                **COMPLETIONS_API_PARAMS
            )

    return response["choices"][0]["text"].strip(" \n"), context


@st.cache_data
def load_data():
    """ Utility function for loading the penguins dataset as a dataframe."""
    # df = sns.load_dataset('penguins')

    with open('embeddings.pkl', 'rb') as f:
        doc_embeddings = pickle.load(f)
    df = pd.read_csv('paginas.txt')
    return df, doc_embeddings

# load dataset
df, doc_embeddings = load_data()

#---------------------------------------------------------------
# Ask a question and search for the answer

font_size = "20px"
background_color = "#F9F9F9"
text_color = "#00f900"

st.markdown(f"""
    <style>
        input {{
            font-size: {font_size};
            background-color: {background_color};
            color: {text_color};
        }}
    </style>
""", unsafe_allow_html=True)

question = st.text_input(
    "Ask a question:",
     value="",
    max_chars=None,
    key=None,
    type="default",
)
if question:
    answer, context = answer_query_with_context(question, df, doc_embeddings)
    # Replace newline characters with line breaks
    answer_with_line_breaks = answer.replace('\n', '<br>')
    # Display the answer as a paragraph of text with line breaks
    st.markdown(f"<p>{answer_with_line_breaks}</p>", unsafe_allow_html=True)
    st.markdown("""
    <style>
    p {
        font-size: 18px;
        line-height: 1.5;
        text-align: justify;
    }
    </style>
""", unsafe_allow_html=True)


    

