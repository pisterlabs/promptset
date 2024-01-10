# from import Flask, render_template, request, jsonify
import pinecone
import os
import openai
import uuid

os.environ["OPENAI_API_KEY"] = "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"
os.environ["PINECONE_API_KEY"] = "eea250b5-8ade-4981-ab53-626dc466ac53"

def generate_embedding(prompt):
    openai.api_key = os.environ["OPENAI_API_KEY"]
    # openai_client = openai.OpenAI()
    prompt_vector = openai.Embedding.create(
        input=prompt,
        model="text-embedding-ada-002"
    ).data[0].embedding

    return prompt_vector

def search_results(prompt, prompt_type=None, domain=None):
    # user_prompt = request.form.get('user_prompt')
    # prompttype = request.form.get('prompttype')
    # domain = request.form.get('domain')
    print(prompt_type, domain)
    print(prompt)
    filter = {}
    if prompt_type:
      filter["prompt_type"] = prompt_type
    if domain:
      filter["domain"] = domain
    if prompt:
        pinecone.init(api_key=os.environ["PINECONE_API_KEY"], environment="gcp-starter")
        index = pinecone.Index(index_name="prompt-library")
        embedding = generate_embedding(prompt)
        response = index.query(
            vector=embedding,
            top_k=2,
            include_metadata=True
        )
        if response:
            id_values = [(item['id'], item['score']) for item in response['matches']]
            print(f'id_values: {id_values}')
            return id_values
        else:
            print("No matching response found.")

def add_prompt(prompt, prompt_type, domain):
    # prompt = request.form.get('prompt')
    # prompttype = request.form.get('prompttype')
    # domain = request.form.get('domain')
    if prompt:
        embedding = generate_embedding(prompt)

        pinecone.init(api_key=os.environ["PINECONE_API_KEY"], environment="gcp-starter")
        index = pinecone.Index(index_name="prompt-library")
        metadata = {"prompttype": prompt_type, "domain": domain}
        vector_id = str(uuid.uuid4())
        pinecone_vector = (vector_id, embedding, metadata)
        print(vector_id, prompt, metadata)
        upsert_count = index.upsert(vectors=[pinecone_vector])['upserted_count']
        index_stats = index.describe_index_stats()
        response = f"Your prompt has been added! Total prompt count {index_stats['total_vector_count']}"
        if upsert_count:
            print(response)
    else:
        return print("Please provide a user prompt.")