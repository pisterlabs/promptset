import os
import pandas as pd
import elasticsearch
from semantic_search_service import SemanticSearchService
from elasticsearch.helpers import bulk

class Indexer:
    def __init__(self, es_client, get_embedding):
        self.es_client = es_client
        self.get_embedding = get_embedding

    def index_data(self, data, index_name):
        # Create the index with a mapping for dense_vector if it doesn't exist
        if not self.es_client.indices.exists(index=index_name):
            self.es_client.indices.create(
                index=index_name,
                body={
                    "mappings": {
                        "properties": {
                            "Title": {
                                "type": "text"
                            },
                            "Text": {
                                "type": "text"
                            },
                            "Link": {
                                "type": "text"
                            },
                            "Embedding": {
                                "type": "dense_vector",
                                "dims": 1536  # The dimension of OpenAI embeddings
                            }
                        }
                    }
                }
            )

        # Generate the list of actions to pass to the `bulk` method
        actions = []
        for i, row in data.iterrows():
            text_embedding = self.get_embedding(row["Text"])
            action = {
                "_op_type": "index",
                "_index": index_name,
                "_id": i,
                "_source": {
                    "Title": row["Title"],
                    "Text": row["Text"],
                    "Link": row["Link"],
                    "Embedding": text_embedding
                }
            }
            actions.append(action)

        # Use the `bulk` method to index all the documents in a single request 
        #  to make it significantly faster when dealing with a large number of documents.
        bulk(self.es_client, actions)

        # Refresh the entire index after indexing all the documents
        self.es_client.indices.refresh(index=index_name)


    def index_data_from_file(self, file_name, index_name):
        # Read the data from a CSV file
        df = pd.read_csv(file_name)
        # Index the data with embeddings
        self.index_data(df, index_name)

    def load_index(self, file_name, index_name):
        # If the index doesn't exist, create it by indexing the data from the CSV file
        if not self.es_client.indices.exists(index=index_name):
            self.index_data_from_file(file_name, index_name)

# # Define a method to get embeddings from OpenAI
    # def get_embedding(self, text, model="text-embedding-ada-002"):
    #     response = openai.Embedding.create(
    #         input=text,
    #         model=model
    #     )
    #     embeddings = response['data'][0]['embedding']
    #     return embeddings

def main():
    # Set up OpenAI API key and Elasticsearch client
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    # Check if the API key is present
    if not openai_api_key:
        raise ValueError("The OPENAI_API_KEY environment variable is not set. Please set it before running the script.")
    
    # Create an instance of the Elasticsearch client
    es_client = elasticsearch.Elasticsearch(hosts=["http://localhost:9200"])

    # Create an instance of the semantic search service class
    semantic_search_service = SemanticSearchService(openai_api_key, es_client)

    # Create an instance of the Indexer class
    indexer = Indexer(es_client, semantic_search_service.get_embedding)

    # Load (and index if necessary) the data
    index_name = "sample"
    file_name = 'annotators.csv'

    # delete index if exists
    if es_client.indices.exists(index=index_name):
        es_client.indices.delete(index=index_name)

    indexer.load_index(file_name, index_name)

if __name__ == "__main__":
    main()
