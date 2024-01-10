from elasticsearch import Elasticsearch
import streamlit as st
import openai

es_username = st.secrets["es_username"]
es_password = st.secrets["es_password"]
es_cloudid = st.secrets["es_cloudid"]

es_index = st.secrets["es_index2"]

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
                    "title": {
                        "query": query_text,
                        "boost": 1
                    }
                }
            }],
            "filter": [{
                "exists": {
                    "field": "description-body"
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

    title = resp['hits']['hits'][0]['_source']['title']
    body = resp['hits']['hits'][0]['_source']['description-body']
    url = resp['hits']['hits'][0]['_source']['url']

    return title, body, url

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
                                            messages=[{"role": "system", "content": "You are a cyber security analyst with decades of experience, a SANS instructor and defend networks from nation state actors."}, {"role": "user", "content": truncated_prompt}])

    return response["choices"][0]["message"]["content"]

st.set_page_config(layout="wide")
st.title("MITRE GPT")
st.subheader("This is a concept conversational Q/A chatbot using Tactics and Techniques from MITRE ATT&CK® knowledge base.")
st.write("© 2023 The MITRE Corporation. This work is reproduced and distributed with the permission of The MITRE Corporation. https://attack.mitre.org")

# Main chat form
with st.form("chat_form"):
    query = st.text_input("What is your security concern today?")
    submit_button = st.form_submit_button("Chat with MITREGPT")

# Generate and display response on form submission
negResponse = "I'm unable to answer the question based on the information I have from your docs."

if submit_button:
    title, body, url = search(es_index, query)

    st.subheader(title)
    st.write(f"MITRE Content as : {body}")
    st.divider()
    
    analyst_prompt = f"Using your years of experience as a blue team cyber defender, can you help me with this question: {query}\nUsing only the following information from mitre attack framework and your experience: {body}\nPlease structure the response to a cyber security analyst with a summary and 3 bullet points, additionally provide another 3 points from your analyst experience that is not found in the above\nIf the answer is not contained in the supplied doc reply '{negResponse}' and nothing else"

    # ciso_prompt = f"Using your years of experience as a blue team cyber defender, can you help me with this question: {query}\nUsing only the following information from mitre attack framework and your experience: {body}\nPlease structure the response to a cyber security analyst with a summary and 5 bullet points, additionally provide another 3 points from your analyst experience that is not found in the above\nIf the answer is not contained in the supplied doc reply '{negResponse}' and nothing else"

    analyst_response = chat_gpt(analyst_prompt)
    # ciso_response = chat_gpt(ciso_prompt)

    if negResponse in analyst_response:
        st.write(f"MitreGPT: {analyst_response.strip()}")
        st.divider()
    else:
        st.write(f"MitreGPT Response for Cyber Analyst: {analyst_response.strip()}")
        st.divider()

    # if negResponse in ciso_response:
    #     st.write(f"MitreGPT: {ciso_response.strip()}")
    #     st.divider()
    # else:
    #     st.write(f"MitreGPT Response: {ciso_response.strip()}\n\nDocs: {url}")
    #     st.divider()

    st.write(f"MITRE Content: {body}")
    st.write(url)