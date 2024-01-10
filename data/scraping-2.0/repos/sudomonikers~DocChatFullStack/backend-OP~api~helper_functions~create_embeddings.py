import openai
import os 
from typing import List

openai.api_key = os.getenv('OPENAI_API_KEY')

def get_openai_embeddings(sentences: List[str], model = "text-embedding-ada-002") -> List[List[float]]:
    embeddings = []
    response = openai.Embedding.create(model=model, input=sentences)
    for data in response.data:
        embeddings.append(data["embedding"])
    return embeddings

if __name__ == "__main__":
    #example usage
    print(get_openai_embeddings(["What is the best way to get a job at OpenAI?"]))