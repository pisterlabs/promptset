import pinecone
import openai
import uuid
import constants as c   

pinecone.init(api_key=c.PINECONE_AUTH, environment='gcp-starter')
index = pinecone.Index(index_name='prompt-library')

def generate_embedding(prompt):
    prompt_vector = openai.Embedding.create(
        input=prompt,
        model='text-embedding-ada-002'
    ).data[0].embedding

    return prompt_vector

def search_results(prompt, prompt_type=None, domain=None):
    filter = {}
    if prompt_type:
      filter['prompt_type'] = prompt_type
    if domain:
      filter['domain'] = domain
    if prompt:
        
        embedding = generate_embedding(prompt)
        response = index.query(
            vector=embedding,
            top_k=2,
            include_metadata=True
        )
        if response:
            id_values = [(item['id'], item['score']) for item in response['matches']]
            return id_values
        else:
            return 0

def add_prompt(prompt, prompt_type, domain):
    if prompt:
        embedding = generate_embedding(prompt)

        metadata = {'prompttype': prompt_type, 'domain': domain}
        vector_id = str(uuid.uuid4())
        pinecone_vector = (vector_id, embedding, metadata)
        index.upsert(vectors=[pinecone_vector])['upserted_count']
    else:
        return print('Please provide a user prompt.')