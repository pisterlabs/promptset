import pandas as pd
import openai
from openai.embeddings_utils import distances_from_embeddings
import numpy as np

# feature: robot mode like 
# assistant: answer question from the knowledge base
# sales: try to sell product

# feature: can have conversation

# clean up
# think about how to clean up each part of coding

openai.api_key_path = 'apikey.txt'
question = input('you: ')
response = openai.Embedding.create(input=question, engine='text-embedding-ada-002')
print(response['usage']['total_tokens'])
q_embeddings = response['data'][0]['embedding']
df = pd.read_csv('text/www.centific.com.csv')
df['embeddings'] = df['embeddings'].apply(eval).apply(np.array)
df['distances'] = distances_from_embeddings(
    q_embeddings, df['embeddings'].values, distance_metric='cosine')
df = df.sort_values('distances', ascending=True)
print(df.head()[['name', 'tokens']])

print(f'context name: {df.iloc[0]["name"]}')
context = df.iloc[0]["text"]
# print(context)

messages = [
    # {"role": "system", "content": f"You are a sales who try to promote related product and solution when possible. Answer the question based on the context below: \"{context}\"."},
    {"role": "system", "content": f"You are a helpful assistant. Answer the question based on the context below: \"{context}\""},
    {"role": "user", "content": f"{question}"},
]
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=messages
)
print(response['usage']['total_tokens'])
print(response['choices'][0]['message']['content'])
messages.append({
    "role": response['choices'][0]['message']['role'],
    "content": response['choices'][0]['message']['content']
})
