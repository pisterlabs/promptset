from config import *
import openai
import time
from openai.embeddings_utils import get_embedding
import pinecone

openai.api_key = openai_apikey
pinecone.init(api_key=pinecone_apikey, environment=pinecone_envirnment)
index = pinecone.Index("tutorial")



while True:
    my_inp = input("Enter your query: ")
    if len(my_inp) == 0:
        break

    my_inp_embed = get_embedding(my_inp, engine="text-embedding-ada-002")
    
    res = index.query(
        vector=my_inp_embed,
        top_k=3,
        include_metadata=True,
        include_values=False
    )

    context = ''
    for i in res['matches']:
        context += i['metadata']['text'] + '\n'

    response = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Data: {}\n\n You will answer according to the provided data".format(context)},
        {"role": "assistant", "content": "Okay got it."},
        {"role": "user", "content": my_inp}
    ]
)
    
    print(response['choices'][0]['message']['content'])
    
    
    
    

