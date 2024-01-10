from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from dotenv import load_dotenv
import os
from supabase import create_client, Client
import numpy as np

load_dotenv()

# Initialize Supabase Client
url = os.environ.get('SUPABASE_URL')
key = os.environ.get('SUPABASE_KEY')
supabase: Client = create_client(url, key)

# Initialize OpenAI Client
opeani_key = url = os.environ.get('OPENAI_KEY')
client = OpenAI(api_key=opeani_key)


response = client.embeddings.create(
    input="text",
    model="text-embedding-ada-002"
)
print(f'vector_search.py / response: {response}')

embedding = np.array(response.data[0].embedding).reshape(1, -1)

print(f'vector_search.py / embedding: {embedding}')


def calculate_similarity(embedding, embeddings_list):
    similarities = cosine_similarity(embedding, embeddings_list)
    return similarities[0]

# Convert the string embedding to numpy array using numpy.fromstring()
def convert_embedding(str_embedding):
    return np.fromstring(str_embedding.strip("[]"), sep=',', dtype=float)

# Get the embeddings from the database, id of the company and the embedding
data = supabase.table("embedings").select("id, embedding").execute().data
print(f'vector_search.py / data: {data}')

# Create a list of embeddings and a list of ids
db_embeddings = np.array([row['embedding'] for row in data])
print(f'vector_search.py / db_embeddings: {db_embeddings}')

# Convert the embeddings to numpy array using the function above (convert_embedding())
db_embeddings = np.array([convert_embedding(row['embedding']) for row in data])
print(f'vector_search.py / db_embeddings: {db_embeddings}')
ids = [row['id'] for row in data]
print(f'vector_search.py / ids: {ids}')

# Calculate the similarity between the user input and the embeddings from the database
similarities = calculate_similarity(embedding, db_embeddings)
print(f'vector_search.py / similarities: {similarities}')

# Get the top 5 similar ids, in descending order of similarity
top_indices = np.argsort(similarities)[::-1][:5]
print(f'vector_search.py / top_indices: {top_indices}')

# Print the top 5 similar ids
print("Top 5 similar IDs in descending order of similarity:")
for index in top_indices:
    print(f"ID: {ids[index]}, Similarity: {similarities[index].round(4)}")
  