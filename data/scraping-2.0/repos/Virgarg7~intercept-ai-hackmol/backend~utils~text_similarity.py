import pandas as pd
import os
import openai , numpy as np
from openai.embeddings_utils import get_embedding, cosine_similarity

apikey = os.getenv('OPENAI_API_KEY')
openai.api_key = apikey


def get_embedding(text, model="text-embedding-ada-002"):
   text = text.replace("\n", " ")
   return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']
 
block_words = ['fuck' ,'bastard' ,'asshole' ,'son of a bitch' ,'bloody hell' ,'dick head' ,'bullocks' ,'cock sucker','mother fucker '
,'pussy','You saved my ass','dickhole' ]


# def checkSimilar(word):
#    Ex = get_embedding(word)
#    ans =0
#    for x in block_words:
#      embed = get_embedding(x)
#      ans = max(ans , cosine_similarity(Ex ,embed)) 
#      print(ans)
#    return ans >0.84


block_words_embed =[]
for x in block_words:
     embed = get_embedding(x)
     block_words_embed.append(embed)


def checkSimilar2(word):
    Ex = get_embedding(word)
    ans =0
    for x in block_words_embed:
      v = cosine_similarity(Ex ,x)
      ans = max(ans , v) 
      print(v)
    return ans > 0.85

print(checkSimilar2("kashmir"))