import os
import json
import torch
from sentence_transformers import SentenceTransformer
import openai
from qdrant_client import QdrantClient
from qdrant_client.http import models

COLLECTION_NAME="mybook"

def get_environment_variables():
    # Retrieve necessary environment variables
    qdrant_host = os.environ.get('QDRANT_HOST')
    OPENAI_API_KEY = os.environ.get('MYOPENAI')
    return qdrant_host, OPENAI_API_KEY

def setup_openai_client(api_key):
    # Set up the OpenAI client
    openai.api_key = api_key

def setup_qdrant_client(qdrant_host):
    # Set up the Qdrant client
    return QdrantClient(qdrant_host)

def choose_device():
    # Choose which device to use for sentence transformers
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    return device

def setup_sentence_transformer(device):
    # Set up sentence transformer
    return SentenceTransformer("msmarco-MiniLM-L-6-v3", device)

def build_prompt(question, references):
    # Function to build prompt from question and references
    # Removed unused strip()
    prompt = f"""
    You're Marcus Aurelius, emperor of Rome. You're giving advice to a friend who has asked you the following question: '{question}'

    You've selected the most relevant passages from your writings to use as source for your answer. Cite them in your answer.

    References:
    """

    references_text = ""

    for i, reference in enumerate(references, start=1):
        text = reference.payload["text"]
        references_text += f"\n[{i}]: {text}"

    prompt += (
        references_text
        + "\nHow to cite a reference: This is a citation [1]. This one too [3]. And this is sentence with many citations [2][3].\nAnswer:"
    )
    return prompt, references_text

def search_similar_docs(question, qdrant_client, retrieval_model):
    query_vector=retrieval_model.encode(question)
    print(f"query_vector length {len(query_vector)} ={query_vector[:2]} ~ {query_vector[-2:]}")
    # Function to find similar documents    
    similar_docs = qdrant_client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=3,
        append_payload=True,
    )
    return similar_docs

def ask(question, qdrant_client, retrieval_model):
    # Function to ask the question
    similar_docs = search_similar_docs(question, qdrant_client, retrieval_model)
    prompt, references = build_prompt(question, similar_docs)
    return (prompt, references)

def ask_and_get_response(question, qdrant_client, retrieval_model):
    # Function to ask question and get response from GPT model
    similar_docs = search_similar_docs(question, qdrant_client, retrieval_model)
    prompt, references = build_prompt(question, similar_docs)

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt},
        ],
        max_tokens=250,
        temperature=0.2,
    )

    return {
        "response": response,
        "references": references,
    }

def main():
    # Main function to orchestrate the flow
    qdrant_host, OPENAI_API_KEY = get_environment_variables()
    setup_openai_client(OPENAI_API_KEY)
    qdrant_client = setup_qdrant_client(qdrant_host)
    device = choose_device()
    retrieval_model = setup_sentence_transformer(device)
    
    text = input("Enter your question: ")
    #response=ask(text, qdrant_client, retrieval_model)
    response=ask_and_get_response(text, qdrant_client, retrieval_model)
    print("\n" + "-" * 40 + "\n")
    print(f"{response}")
    print("\n" + "-" * 40 + "\n")
    #print("copy & paste to chatgpt")

if __name__ == "__main__":
    main()
