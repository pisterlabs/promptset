import openai
import streamlit as st
from utils import get_current_conversation, get_working_collection, get_cohere_client, create_meta_filter, get_openai_key
import cohere
import time
import threading
import os
import requests
import json
from threading import Semaphore

class ConcurrentCallLimiter:
    def __init__(self, max_concurrent_calls):
        self.semaphore = Semaphore(max_concurrent_calls)

    def acquire(self):
        self.semaphore.acquire()

    def release(self):
        self.semaphore.release()

# Initialize the call limiter for 5 concurrent calls
call_limiter_translation = ConcurrentCallLimiter(10)
call_limiter_embedding = ConcurrentCallLimiter(30)

def call_with_timeout_translation(func, args):
    result = [None]
    exception = [None]

    def target():
        try:
            call_limiter_translation.acquire()
            try:
                result[0] = func(*args)
            finally:
                call_limiter_translation.release()
        except Exception as e:
            exception[0] = e

    thread = threading.Thread(target=target)
    thread.start()
    thread.join()
    return result[0], exception[0]

def call_with_timeout_embed(func, args):
    result = [None]
    exception = [None]

    def target():
        try:
            call_limiter_embedding.acquire()
            try:
                result[0] = func(*args)
            finally:
                call_limiter_embedding.release()
        except Exception as e:
            exception[0] = e

    thread = threading.Thread(target=target)
    thread.start()
    thread.join()
    return result[0], exception[0]

def embed_text(text, key):
    api_url = "https://api.openai.com/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json"
    }
    data = {
        "input": text,
        "model": "text-embedding-ada-002",
        "encoding_format": "float"
    }

    passed = False
    for j in range(5):
        try:
            response = requests.post(api_url, headers=headers, json=data)
            if response.status_code == 200:
                passed = True
                break
            elif response.status_code == 429:  # Rate limit error
                time.sleep(2 ** j)
            else:
                break  # Break on other errors
        except Exception as e:
            print(f"Exception occurred: {e}")
            break

    if not passed:
        raise RuntimeError("Failed to create embeddings.")
    
    embedding = response.json()['data'][0]['embedding']
    return embedding

def get_model_response(query, conversation):
    openai.api_base = "https://openrouter.ai/api/v1"
    openai.api_key_path = "openrouter.txt"

    messages = [
        {"role": "system", "content": "You are a virtual knowledge agent who is provided snippets of data from various files. You attempt to fulfill queries based on provided context whenever possible."}
    ]

    if conversation:
        messages = get_current_conversation()

    prompt = {"role": "user", "content": query}

    messages.append(prompt)

    response = openai.ChatCompletion.create(
        #model = "",
        model="openai/gpt-4-1106-preview",
        messages=messages,
        headers={
            "HTTP-Referer": "http://codebase.bogusz.co",
        },
        stream=True,
    )
    
    for chunk in response:
        yield chunk

def get_chunk_classification(query, metadata):
    openai.api_base = "https://openrouter.ai/api/v1"
    openai.api_key_path = "openrouter.txt"

    template_file = "translation_template.txt"
    
    if os.path.exists(template_file):
        with open(template_file, 'r') as file:
            template = file.read()
    else:
        template = """Translate the following code snippet into a natural language.
            Metadata:
            {metadata}

            Snippet:
            {query}
        """

    augmented_prompt = template.format(metadata=metadata, query=query)

    messages = [
        {"role": "system", "content": "You are a virtual knowledge agent who is provided snippets of data from various files. You attempt to fulfill queries based on provided context."},
        {"role": "user", "content": augmented_prompt}
    ]

    response = openai.ChatCompletion.create(
        #model = "",
        model="openai/gpt-3.5-turbo-1106",
        messages=messages,
        headers={
            "HTTP-Referer": "http://codebase.bogusz.co",
        },
    )

    response_content = response.choices[0].message

    openai.api_base = "https://api.openai.com/v1"
    openai.api_key_path = "openai.txt"

    return response_content["content"]

def inject_context(query):
    collection = get_working_collection()

    key = get_openai_key()

    vector, error = call_with_timeout_embed(embed_text, [query, key])
    if error:
        print(f"Error or timeout on first try embedding: {error}. Retrying...")
        vector, error = call_with_timeout_embed(embed_text, [query, key])
        if error:
            print(f"Error or timeout on second try embedding query: {error}.")
            raise Exception("Unable to vectorize query, failed embedding")

    selected_files_context = st.session_state.get('selected_context_files', [])
    if selected_files_context:
        meta_filter = create_meta_filter(selected_files_context)
        results = collection.query(
            query_embeddings=vector,
            n_results=200,
            include=["documents", "metadatas"],
            where=meta_filter
        )
    else:
        results = collection.query(
            query_embeddings=vector,
            n_results=200,
            include=["documents", "metadatas"]
        )

    # Prepare documents for reranking using original document content
    documents_for_reranking = [{'text': doc} for doc in results['documents'][0]]

    co = get_cohere_client()

    # in case we return less than 15 embeddings
    total_docs = len(documents_for_reranking)
    top_n = min(total_docs, 15)

    # Complete rerank call
    reranked_results = co.rerank(
        query=query,
        documents=documents_for_reranking,
        model="rerank-english-v2.0",
        top_n=top_n
    )

    structured_context = f"""
        **Query for Analysis:**
        {query}

        **Provided Context:**
        The following sections contain mixed types of data from various sources for context. Each section is clearly marked with its source origin and type. The content includes both plain text and formatted code blocks, as indicated.

        **Contexts:**

        """

    for rank in reranked_results.results:
        doc_index = rank.index
        # Substitute translation metadata into document text if it exists
        meta = results['metadatas'][0][doc_index]
        content = meta.get('translation') if meta.get('translation') and len(meta['translation']) > 5 else documents_for_reranking[doc_index]['text']
        structured_context += f"{rank.index+1}. **[Source: {meta['source']}, Type: {meta['type']}, Language: {meta['language']}, Filename: {meta['filename']}]**\n```\n{content}\n```\n\n"

    return structured_context

def begin_conversation():
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask your question"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        query = inject_context(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            for partial_response in get_model_response(query, False):
                full_response += (partial_response.choices[0].delta["content"] or "")
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})