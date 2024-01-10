import os
import cohere
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
co = cohere.Client(os.environ['COHERE_API_KEY'])

def encode_text(text : str):
    return model.encode(text)

def summarize_documents(text : str):
    response = co.summarize(
        text=text,
        model='command',
        length='medium',
        extractiveness='low',
        temperature=5
    )

    summary = response.summary
    return summary