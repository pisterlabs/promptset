from document_chunker import DocChunker
import pinecone
import numpy as np
import spacy
from PyPDF2 import PdfReader
import nltk
import sys
import json
import os
import openai
from scipy.spatial import distance
import plotly.express as px
from sklearn.cluster import KMeans
from umap import UMAP

#GET THE VECTOR FROM JAVASCRIPT
pinecone.init(api_key="b0ec4895-ab56-43b3-baf0-a404a9e28e20", environment="gcp-starter")
openai.api_key = "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"
if not("data-embeddings" in pinecone.list_indexes()):
    pinecone.create_index("data-embeddings", dimension=1536, metric="euclidean")
this_table = pinecone.Index("data-embeddings")

def getQueryAns(this_question):
    response = openai.Embedding.create(model= "text-embedding-ada-002", input=[this_question])
    this_ans = this_table.query(vector=response["data"][0]["embedding"], top_k=2, include_values=True)
    prompt = "You are a Certified Public Accountant who was asked: " + this_question + ". Please answer using this as context:" + this_ans['matches'][0]['id'] + ", " + this_ans['matches'][1]['id']
    response = openai.Completion.create(
    engine="davinci",
    prompt=prompt,
    max_tokens=500
    )

    return response.choices[0].text.strip()

def main(this_question):
    with open('output_query.json', 'w') as f:  # Write the results to a file
        json.dump(getQueryAns(this_question), f)

if __name__ == "__main__":
    main(sys.argv[1])