import os
import openai
import elasticsearch


class SemanticSearchService:

    def __init__(self, openai_api_key, es_client):
        self.openai_api_key = openai_api_key
        self.es_client = es_client

    def get_embedding(self, text, model="text-embedding-ada-002"):
        max_tokens = 8000
        text_parts = [text[i: i + max_tokens] for i in range(0, len(text), max_tokens)]
        embeddings = []
        for part in text_parts:
            response = openai.Embedding.create(
                input=part,
                model=model
            )
            embeddings.append(response['data'][0]['embedding'])
        avg_embedding = [sum(col) / len(col) for col in zip(*embeddings)]
        return avg_embedding

    def semantic_search(self, query, index_name, top_k=5):
        query_embedding = self.get_embedding(query)
        search_query = {
            "query": {
                "script_score": {
                    "query": {
                        "match_all": {}
                    },
                    "script": {
                        "source": "(cosineSimilarity(params.query_embedding, 'Embedding') + 1.0) / 2.0",
                        "params": {
                            "query_embedding": query_embedding
                        }
                    }
                }
            },
            "size": top_k 
        }
        response = self.es_client.search(
            index=index_name,
            body=search_query
        )
        hits = response["hits"]["hits"]
        results = [(hit["_score"], hit["_source"]["Title"], hit["_source"]["Link"],hit["_source"]["Text"]) for hit in hits]
        return results

def main():
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("The OPENAI_API_KEY environment variable is not set. Please set it before running the script.")

    es_client = elasticsearch.Elasticsearch(hosts=["http://localhost:9200"])

    semantic_search_service = SemanticSearchService(openai_api_key, es_client)

    index_name = "sample"
    search_text = "How to convert token to chunk?"
    results = semantic_search_service.semantic_search(search_text, index_name)

    print("\nWriting to output.txt file...")
    for result in results:
        with open('output.txt', 'a') as f:
            f.write(f"Similarity: {result[0]} \nTitle: {result[1]} \nLink: {result[2]}\n\n")

    print("Completed!")
    
if __name__ == "__main__":
    main()
