import os
import pandas as pd
import json
from openai.embeddings_utils import get_embedding, cosine_similarity
import openai
openai.api_key = os.environ.get('OPENAI_KEY')


def load():
   indices = []
   cols = ['embeddings']
   data = []
   for item in os.listdir('embeddings'):
      if item.endswith('.json'):
         with open(os.path.join('embeddings',item)) as f:
             json_str = f.read()
             embedding = json.loads(json_str) 
             xkcd_id = item.replace('.json','')
             indices.append(xkcd_id)
             data.append({'embeddings': embedding['data'][0]['embedding']})
          
   df = pd.DataFrame(data, index = indices, columns = cols)  
   return df

# search through the xkcds
def search_xkcd(reply, n=5, pprint=True):
    reply_embedding = get_embedding(
        reply,
        engine="text-embedding-ada-002"
    )
    df["similarity"] = df.embeddings.apply(lambda x: cosine_similarity(x, reply_embedding))

    print(df['similarity'])
    results = (
        df.sort_values("similarity", ascending=False)
        .head(n)
    )
    if pprint:
        for r in results:
            print(r)
            print()
    
    return results

print('loading data')
df = load()

if __name__ == '__main__':
    search_xkcd('compiling!')
    #results = search_xkcd(df, 'my code is compiling')
    #print(results.index.values.astype(str)[0])


