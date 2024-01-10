from openai import OpenAI

client = OpenAI()

embedding_call = {
    'input': "Once upon a time",
    'model': "text-embedding-ada-002"
}

embeddings = client.embeddings.create(**embedding_call)

len_embedding = len(embeddings.data[0].embedding)

if len_embedding is not None:
    print(f'The embedding has {len_embedding} dimensions')
else:
    print('you done goofed')