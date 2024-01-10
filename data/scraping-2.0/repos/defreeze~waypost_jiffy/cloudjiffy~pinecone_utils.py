import openai
import pinecone
import numpy as np


def store_embeddings_in_pinecone(data, openai_api_key, pinecone_api_key):
    # Step 1: Initialize OpenAI API
    openai.api_key = openai_api_key

    # Step 2: Initialize Pinecone
    # pinecone.init(api_key=pinecone_api_key)
    pinecone.init(
        api_key="bb625a20-105e-4675-a446-5877a8decb16", environment="eu-west4-gcp"
    )

    # Define the index
    index_name = "cloudjiffy1536"
    index = pinecone.Index(index_name=index_name)
    # pinecone.create_index(index_name="url_embeddings")
    count = 1
    for url, text in data.items():
        # choose an embedding
        model_id = "text-similarity-davinci-001"

        # compute the embedding of the text
        embedding = openai.Embedding.create(input=text, model=model_id)["data"][0][
            "embedding"
        ]
        if count == 1:
            print(f"embedding: {embedding}")
        count += 1

        # Convert the embedding to a list to store in Pinecone
        # embedding_list = embedding.tolist()
        print(f"Embedding type: {type(embedding)}, shape: {np.array(embedding).shape}")

        embedding_list = np.array(embedding).flatten().tolist()

        # Store the embedding with metadata (URL) in Pinecone vector store
        index_name = "cloudjiffy1536"
        index = pinecone.Index(index_name)

        index.upsert(
            ids=[url],
            vectors=[embedding],
            metadata=[url],
        )

    pinecone.deinit()
