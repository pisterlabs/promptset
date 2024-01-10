import pandas as pd
import tiktoken
from openai.embeddings_utils import get_embedding
import numpy as np
from openai import OpenAI

# embedding model parameters
embedding_model = "text-embedding-ada-002"
embedding_encoding = "cl100k_base"  # this the encoding for text-embedding-ada-002
max_tokens = 8000  # the maximum for text-embedding-ada-002 is 8191
# Create the vector table
# Initialize the OpenAI class
openai = OpenAI()

#read csv file
df = pd.read_csv('./models/exchange.csv')

#create embedded table
table = df.pivot_table(index='_id', columns='country', values=['market_name', 'mic', 'operating_mic', 'website'])

#print embedded table
#print(table)

#store vector data and its text equivalent
vector_data = dict()
for index, row in df.iterrows():
    vector_data[row['_id']] = {
        'market_name': row['market_name'],
        'mic': row['mic'],
        'operating_mic': row['operating_mic'],
        'website': row['website']
    }
    
#print(vector_data)

#create table to store vector data and text data
df_vectorData = pd.DataFrame.from_dict(vector_data, orient='index')

#export the table as csv
df_vectorData.to_csv('./vector_data/exchange_vector_data.csv', index=False)

#perform semantic query
#import vector_data.csv
df_vectorData = pd.read_csv('./vector_data/exchange_vector_data.csv')

#perform semantic query on vector_data.csv
query = df_vectorData[df_vectorData['operating_mic'] == 'AAA'][['market_name', 'website']]

print("Semantic Query Results:")
print(query)

#perform semantic search using OpenAI embeddings
embeddings = get_embedding(embedding_model)

query_embedding = embeddings[query]

print("OpenAI Embeddings Results:")
print(query_embedding)
