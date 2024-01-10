import weaviate
import json
import openai
from .dependencies import get_weaviate_client, get_minio_client, get_openai_api_key
from langchain.llms import OpenAI
from langchain.prompts import CustomQueryGenerationPromptTemplate

# Import get_weaviate_client instead of Pinecone-related imports
class CustomQueryGenerationPromptTemplate(StringPromptTemplate):
    def format(self, user_input: str) -> str:
        prompt = (
            "You are a sophisticated AI capable of generating insightful and "
            "relevant search queries based on a given topic. Transform the "
            "following user input into a series of refined search queries.\n\n"
            f"User Input: {user_input}\n\n"
            "Generated Queries:\n1. "
        )
        return prompt

def execute_rag_fusion_chain(query: str, bucket_name: str):
    weaviate_client = get_weaviate_client()
    minio_client = get_minio_client()
    openai_api_key = get_openai_api_key()

    # 1. Query generation using custom prompt template
    prompt_template = CustomQueryGenerationPromptTemplate()
    formatted_prompt = prompt_template.format(query)
    generated_queries = OpenAI().generate(formatted_prompt)  # Adjust with actual LLM call

    # 2. Vector-based search using Weaviate
    search_results = []
    for generated_query in generated_queries:
        openai.api_key = openai_api_key
        model="text-embedding-ada-002"
        oai_resp = openai.Embedding.create(input=[generated_query], model=model)
        oai_embedding = oai_resp['data'][0]['embedding']
        result = weaviate_client.query.get("Your_Class", ["your_properties"]).with_near_vector({"vector": oai_embedding, "certainty": 0.7}).with_limit(2).do()
        search_results.append(result)

    # 3. Reciprocal Rank Fusion
    combined_results = reciprocal_rank_fusion(search_results)
    related_data = fetch_related_data_from_minio(bucket_name, search_results, minio_client)
    # Combine or process combined_results and related_data as needed
    return combined_results

# 4. Fetch related data from MinIO
def fetch_related_data_from_minio(bucket_name: str, search_results, minio_client):
    # Logic to fetch related data based on search results
    related_data = []
    for result in search_results:
        object_name = result['object_name']  # Adjust based on your data schema
        try:
            data = minio_client.get_object(bucket_name, object_name).read()
            related_data.append(json.loads(data))
        except Exception as e:
            print(f"Error fetching data from MinIO: {e}")
    return related_data
    return combined_results

def reciprocal_rank_fusion(results):
    # Implement the reciprocal rank fusion algorithm
    fused_scores = {}
    k = 60  # Constant for score calculation, adjust as needed

    for rank, docs in enumerate(results):
        for doc in docs:
            doc_id = doc['id']  # Adjust based on your data schema
            if doc_id not in fused_scores:
                fused_scores[doc_id] = 0
            fused_scores[doc_id] += 1 / (rank + 1 + k)

    reranked_results = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    return reranked_results



