import json
from operator import index
import textract
import requests
import pandas as pd
from langchain.text_splitter import TokenTextSplitter,RecursiveCharacterTextSplitter
import torch
from datasets import load_dataset
from sentence_transformers.util import semantic_search

textra = textract.process("domainTags.csv").decode('utf-8').strip()
    
text_splitter = RecursiveCharacterTextSplitter(
                chunk_size = 300,
                chunk_overlap = 30
                )

texts = text_splitter.split_text(textra)
# print(texts)
with open('config.json') as f:
        config_data = json.load(f)

HUGGINFACE_API = config_data['HUGGINFACE_API']
model_id = "sentence-transformers/all-MiniLM-L6-v2"
hf_token = HUGGINFACE_API
api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_id}"
headers = {"Authorization": f"Bearer {hf_token}"}
def embedding(texts):
    response = requests.post(api_url, headers=headers, json={"inputs": texts, "options":{"wait_for_model":True}})
    return response.json()

# output = embedding_query(chunked_pdf_content[116])
out = embedding(texts)
print(out)
index = range(1, len(out) + 1)

embeddings = pd.DataFrame(out)
embeddings.to_csv("embeddings.csv",index = "False")




faqs_embeddings = load_dataset('VNS-KO/dataset')
dataset_embeddings = torch.from_numpy(faqs_embeddings["train"].to_pandas().to_numpy()).to(torch.float)
question = ["3d technology?"]
output = embedding(question)

query_embeddings = torch.FloatTensor(output)

search_results = semantic_search(query_embeddings, dataset_embeddings, top_k=5)

# Display the top k chunks
for i, result in enumerate(search_results[0]):
    chunk_index = result['corpus_id'] + 1  # Adding 1 to match your indexing
    similarity_score = result['score']
    print(f"Chunk {chunk_index}: Similarity Score: {similarity_score:.8f}")
    print(texts[chunk_index - 1])  # Subtracting 1 to match zero-based indexing
    print("----------------------")

