#0.3.1
# Streamlit based AI web ingestion and Q&A application
# The application can ingest data from web and answer questions based on the ingested data using OpenAI GPT-4 model

import os
import uuid
import requests
import openai
import datetime
import numpy as np
import pickle
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

openai_api_key = 'Your-Open-API-Key-Here'
if not openai_api_key:
    raise ValueError("The OpenAI API key has not been provided. Set the OPENAI_API_KEY environment variable.")
openai.api_key = openai_api_key

def chunk_text(text, max_tokens=8000):
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    for word in words:
        if current_length + len(word) + 1 > max_tokens:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            current_length = 0
        current_chunk.append(word)
        current_length += len(word) + 1
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    return chunks

def get_embedding_for_large_text(text):
    chunks = chunk_text(text)
    embeddings = []
    for chunk in chunks:
        response = openai.Embedding.create(input=chunk, model="text-embedding-ada-002")
        embedding = response['data'][0]['embedding']
        embeddings.append(embedding)
    return embeddings

def create_file_name(url, extension='txt'):
    parsed_url = urlparse(url)
    url_path_parts = parsed_url.path.strip('/').split('/')
    last_part = url_path_parts[-1] if url_path_parts else parsed_url.netloc
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    return f"{last_part}-{current_date}.{extension}"

def get_most_similar_text_chunk(question, embeddings_dict):
    question_embedding = get_embedding_for_large_text(question)[0]
    similarity_scores = []
    for text_chunk_embedding in embeddings_dict['embeddings']:
        similarity_scores.append(cosine_similarity([question_embedding], [text_chunk_embedding])[0][0])
    most_similar_index = np.argmax(similarity_scores)
    return embeddings_dict['text_chunks'][most_similar_index]

def generate_response(question, embeddings_dict):
    similar_text_chunk = get_most_similar_text_chunk(question, embeddings_dict)
    user_prompt = 'Here is the info from the text: {content}'.format(content=similar_text_chunk, question=question)
    messages = [
        {"role": "system", "content": "You are a knowledgeable assistant."},
        {"role": "user", "content": user_prompt}
    ]
    try:
        response = openai.ChatCompletion.create(model="gpt-4", messages=messages)
        return response['choices'][0]['message']['content']
    except Exception as e:
        return str(e)

def extract_and_save_urls(html_content, file):
    soup = BeautifulSoup(html_content, 'html.parser')
    for link in soup.find_all('a'):
        url = link.get('href')
        if url:
            file.write(url + '\n')

def save_embeddings_to_file(embeddings_dict, file_name):
    with open(file_name, 'wb') as file:
        pickle.dump(embeddings_dict, file)

def load_embeddings_from_file(file_name):
    with open(file_name, 'rb') as file:
        return pickle.load(file)

st.title("Multipurpose Crawler V.0.3s ")
st.write("Please enter the URLs to scrape. Click 'Ingest' when done.")

embeddings_dict = {}

urls = st.text_area("Enter URLs to scrape/ingest or 'done' to finish:")
urls = urls.split("\n")

for url in urls:
    url = url.strip()
    if url:
        response = requests.get(url)
        text = response.text
        file_name = create_file_name(url)
        
        with open(file_name, 'w') as file:
            file.write(text)
            extract_and_save_urls(text, file)

        embeddings = get_embedding_for_large_text(text)
        chunks = chunk_text(text)
        embeddings_file_name = create_file_name(url, extension='pkl')
        embeddings_dict[embeddings_file_name] = {'text_chunks': chunks, 'embeddings': embeddings}
        save_embeddings_to_file(embeddings_dict, embeddings_file_name)

st.write(f"Results are stored in the directory: {os.getcwd()}")

# Initialize session state key if it doesn't exist
if 'question_key' not in st.session_state:
    st.session_state['question_key'] = str(uuid.uuid4())

question = st.text_input("Enter a question to ask OpenAI API based on ingested data, or type 'exit' to quit: ", key='question_key')

if question.lower() != 'exit':
    for embeddings_file_name in embeddings_dict.keys():
        response = generate_response(question, embeddings_dict[embeddings_file_name])
        st.write(response)
