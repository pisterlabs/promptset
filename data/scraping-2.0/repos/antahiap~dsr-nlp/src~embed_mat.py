from transformers import BertTokenizer, BertModel
import openai
import os
from sklearn.metrics.pairwise import cosine_similarity
import glob

from dotenv import load_dotenv
load_dotenv() # Load environment variables from .env file

# Get the API key from the environment
openai.api_key = os.getenv("OPENAI_API_KEY")

files = glob.glob('data/mat/MAT_*.txt')
file_contents = []

for file_path in files[:10]:
    
    # Read the content of the text file
    with open(file_path, "r") as file:
        text = file.read()
        file_contents.append(text)


for content in file_contents:
    inputs = tokenizer(content, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        output = model(**inputs)
    embedding = output.pooler_output
    embeddings.append(embedding)

    # Get the embedding from the API
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"
    )
    embeddings = response['data'][0]['embedding']

    input(embeddings)
