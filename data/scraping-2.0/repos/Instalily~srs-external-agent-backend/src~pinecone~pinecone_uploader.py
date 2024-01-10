import os
from dotenv import load_dotenv
import pandas as pd
from openai import OpenAI
import pinecone
from tqdm.auto import tqdm
import time
import uuid
from src.pinecone.chunks import chunk
import numpy as np

load_dotenv()

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
EMBED_MODEL_NAME = 'text-embedding-ada-002'
YOUR_ENV = "gcp-starter"
INDEX_NAME = "mfw-context"
EMBEDDING_DIMENSION = 1536

client = OpenAI()

class DataProcessing:
    def __init__(self, tokenizer):
        self.pinecone = pinecone.init(api_key=PINECONE_API_KEY, environment=YOUR_ENV)
        
        if INDEX_NAME not in pinecone.list_indexes():
            pinecone.create_index(name=INDEX_NAME, metric='cosine', dimension=EMBEDDING_DIMENSION)
            time.sleep(5)

        self.index = pinecone.Index(INDEX_NAME)
        self.tokenizer = tokenizer

    def embed_documents(self, documents):
        embeddings = []
        for doc in documents:
            response = client.embeddings.create(input=doc, model="text-embedding-ada-002")
            embedding = response.data[0].embedding

            # Normalize the embedding for cosine similarity
            norm = np.linalg.norm(embedding)
            normalized_embedding = embedding / norm if norm > 0 else embedding
            embeddings.append(normalized_embedding.tolist())
        return embeddings

    def process_and_upload_data(self, batch_size=1):
        for i in tqdm(range(0, len(chunk), batch_size)):
            i_end = min(len(chunk), i + batch_size)
            batch = chunk[i:i_end]

            metadatas = [{'text': record['text']} for record in batch]
            documents = [record['text'] for record in batch]
            ids = [str(uuid.uuid4()) for _ in batch]

            embeds = self.embed_documents(documents)

            self.index.upsert(vectors=list(zip(ids, embeds, metadatas)))

    # TODO: tokenize text to truncate to a certain length
    def query_results(self, text, max_tokens=10):
        try:
            response = client.embeddings.create(input=text, model="text-embedding-ada-002")
            embedding = response.data[0].embedding
            
            norm = np.linalg.norm(embedding)
            normalized_embedding = embedding / norm if norm > 0 else embedding
            query_vector = normalized_embedding.tolist()

            # Perform the query
            query_results = self.index.query(query_vector, top_k=1,include_metadata=True)

            # Process each match to limit the number of tokens
            limited_output = []
            for match in query_results['matches']:
                match_text = match['metadata']['text']
                tokens =self.tokenizer.tokenize(match_text)
                truncated_tokens = tokens[:max_tokens]
                truncated_text = self.tokenizer.convert_tokens_to_string(truncated_tokens)
                limited_output.append(truncated_text)

            return limited_output
        except Exception as e:
            print("ERROR at pinecone uploader query results", e)

#if __name__ == "__main__":
#    data_processor = DataProcessing()
#    #data_processor.process_and_upload_data()
#    query_text = "Nicholas Miller"
#    results = data_processor.query_results(query_text)
#    print(results)