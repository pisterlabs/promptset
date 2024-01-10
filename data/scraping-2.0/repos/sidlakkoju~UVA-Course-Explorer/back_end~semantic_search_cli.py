# CLI to perform semantic search on UVA courses
from openai.embeddings_utils import get_embedding, cosine_similarity
import pandas as pd
from config import openai_key
import openai
import numpy as np


openai.api_key = openai_key


def get_embedding(text, model="text-embedding-ada-002"):
   text = text.replace("\n", " ")
   return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']



def search_reviews(query, n=3, pprint=True):
   df = pd.read_csv('static_data_with_emebddings.csv')
   df['ada_embedding'] = df.ada_embedding.apply(eval).apply(np.array)

   query_embedding = get_embedding(query, model='text-embedding-ada-002')
   df['similarities'] = df.ada_embedding.apply(lambda x: cosine_similarity(x, query_embedding))
   res = df.sort_values('similarities', ascending=False).head(n)
   return res


def main():
   while True:
        query = input('Enter a query: ')
        res = search_reviews(query, n=3)
        for i in range(3):
            print(f"Course Title: {res.iloc[i]['descr']}")
            print(f"Description: {res.iloc[i]['description']}")
            print(f"Similarity Score: {res.iloc[i]['similarities']}")
            print("----------------------------------------")
        print("\n"*3)



if __name__ == '__main__':
    main()