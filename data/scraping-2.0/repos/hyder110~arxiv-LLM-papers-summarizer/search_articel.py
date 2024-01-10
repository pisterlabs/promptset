import pandas as pd
import numpy as np
import ast
import openai

def convert_embedding(embedding_list):
    embedding_list=ast.literal_eval(embedding_list)
    return np.array(embedding_list,dtype=np.float323)

def search_similar_article(query,df):
    query_embedding=openai.Embedding.create(input=query,model="text-embedding-ada-002")["data"][0]["embedding"]
    df["embedding"]=df["embedding"].apply(convert_embedding)
    simialr_article=np.array([np.dot(query_embedding,x) for x in df['embedding']])
    return df.iloc[simialr_article.argmax()]
df=pd.read_csv('summary_embeddings.csv')

query=input("Enter your query: ")
result=search_similar_article(query,df)

print(f"Most similar article is:\n {result['summary']}\n")