from elasticsearch import Elasticsearch
import streamlit as st
import openai

es_username = st.secrets["es_username"]
es_password = st.secrets["es_password"]
es_cloudid = st.secrets["es_cloudid"]

es_index = st.secrets["es_index3"]

openai.api_key = st.secrets["openai_api_key"]

model = "gpt-3.5-turbo-0301"

def es_connect(cid, user, passwd):
    es = Elasticsearch(cloud_id=cid, basic_auth=(user, passwd))
    return es

def search(index, query_text):
    cid = es_cloudid
    cp = es_password
    cu = es_username
    es = es_connect(cid, cu, cp)

    # Elasticsearch query (BM25) and kNN configuration for hybrid search
    query = {
        "bool": {
            "must": [{
                "match": {
                    "body_content": {
                        "query": query_text,
                        "boost": 1
                    }
                }
            }],
            "filter": [{
                "exists": {
                    "field": "meta_description"
                }
            }]
        }
    }

    knn = {
        "field": "title-vector",
        "k": 1,
        "num_candidates": 20,
        "query_vector_builder": {
            "text_embedding": {
                "model_id": "sentence-transformers__all-distilroberta-v1",
                "model_text": query_text
            }
        },
        "boost": 24
    }

    resp = es.search(index=index,
                     query=query,
                     knn=knn)

    body = resp['hits']['hits'][0]['_source']['body_content']
    url = resp['hits']['hits'][0]['_source']['url']

    return body, url

def truncate_text(text, max_tokens):
    tokens = text.split()
    if len(tokens) <= max_tokens:
        return text

    return ' '.join(tokens[:max_tokens])

# Generate a response from ChatGPT based on the given prompt
def chat_gpt(prompt, model="gpt-3.5-turbo", max_tokens=1024, max_context_tokens=4000, safety_margin=5):
    # Truncate the prompt content to fit within the model's context length
    truncated_prompt = truncate_text(prompt, max_context_tokens - max_tokens - safety_margin)

    response = openai.ChatCompletion.create(model=model,
                                            messages=[{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": truncated_prompt}])

    return response["choices"][0]["message"]["content"]


st.title("Elastic GPT")

# Main chat form
with st.form("chat_form"):
    query = st.text_input("You: ")
    submit_button = st.form_submit_button("Send")

# Generate and display response on form submission
negResponse = "I'm unable to answer the question based on the information I have from Elastic Docs."
if submit_button:
    body, url = search(es_index, query)
    prompt = f"Answer this question: {query}\nUsing only the information from your docs: {body}\nPlease structure the response with a high level summary and 5 bullet points\nIf the answer is not contained in the supplied doc reply '{negResponse}' and nothing else"
    answer = chat_gpt(prompt)
    
    if negResponse in answer:
        st.write(f"ChatGPT: {answer.strip()}")
    else:
        st.write(f"ChatGPT Response: {answer.strip()}\n\nDocs: {url}")
        st.divider()
        st.write(f"Body Content: {body.strip()}")