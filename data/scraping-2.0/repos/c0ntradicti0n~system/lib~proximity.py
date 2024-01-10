import os
import pickle

import redis
import torch
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from sentence_transformers import SentenceTransformer


class CustomEmbeddingWrapper:
    def __init__(self, sentence_model_name, custom_model, host=None, port=6379, db=0):
        # Initialize the Sentence Transformer model
        self.sentence_embedder = SentenceTransformer(sentence_model_name)

        # Custom model that takes sentence embeddings as input
        self.custom_model = custom_model

        # Connect to Redis
        self.redis_client = redis.StrictRedis(
            host=host if host else os.environ.get("REDIS_HOST", "redis"),
            port=port,
            db=db,
        )

    def embed_documents(self, texts):
        embeddings = []

        for text in texts:
            key = text  # Or use a hash of the text

            # Check Redis cache
            serialized_embedding = self.redis_client.get(key)

            if serialized_embedding:
                embedding = pickle.loads(serialized_embedding)
            else:
                # Generate embedding using Sentence Transformer
                embedding = self.sentence_embedder.encode([text])[0]
                # Cache the embedding
                self.redis_client.set(key, pickle.dumps(embedding))

            # Process embedding with the custom model
            custom_embedding = self.custom_model.embed(
                torch.tensor(embedding).unsqueeze(0)
            )

            embeddings.append(custom_embedding.tolist()[0][0])

        return embeddings

    def embed_query(self, query):
        # Check Redis cache first
        key = query  # Or use a hash of the query
        serialized_embedding = self.redis_client.get(key)

        if serialized_embedding:
            embedding = pickle.loads(serialized_embedding)
        else:
            # Generate embedding using Sentence Transformer
            embedding = self.sentence_embedder.encode([query])[0]
            # Optionally cache the embedding
            self.redis_client.set(key, pickle.dumps(embedding))

        # Process embedding with the custom model
        custom_embedding = self.custom_model.embed(torch.tensor(embedding).unsqueeze(0))

        return custom_embedding.tolist()[0][0]


def set_up_db_from_model(hash, input_dict, model, config):
    persist_directory = f"states/{hash}_db/"
    if not os.path.exists(persist_directory) or not os.listdir(persist_directory):
        os.makedirs(persist_directory, exist_ok=True)
    sentence_embedding___siamese = CustomEmbeddingWrapper(
        config.embedding_model, model[model.active_model_name]
    )
    text_splitter = CharacterTextSplitter(chunk_size=3000, chunk_overlap=0)

    enumerated_texts = input_dict

    documents = text_splitter.create_documents(
        list(enumerated_texts.values()),
        metadatas=[{"key": key} for key in enumerated_texts.keys()],
    )
    vector_store = Chroma.from_documents(
        embedding=sentence_embedding___siamese,
        documents=documents,
        collection_name=hash,
        # persist_directory=persist_directory,
    )
    return vector_store
