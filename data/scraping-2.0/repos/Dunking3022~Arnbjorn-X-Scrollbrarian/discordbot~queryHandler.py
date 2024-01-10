import asyncio
import openai
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import pickle

logs = open("logs.txt","a")

openai.api_key = NULL
with open("../pickledSkyrimTextChunks.pkl", 'rb') as pickledChunks:
  chunks = pickle.load(pickledChunks)
pickled_df = pd.read_pickle('../pickledSkyrimTextEmbedding1.pkl')

lore = '''Arnbjorn was once a librarian of the Arcane University, renowned for his vast knowledge of books and ancient tomes. However, his thirst for knowledge was insatiable, and he yearned to learn the secrets of every book ever written. Desperate to obtain this forbidden knowledge, he made a deal with a powerful Daedric prince, promising to do its bidding in exchange for the knowledge he sought.

The Daedric prince granted Arnbjorn his wish, imbuing him with the knowledge of every book ever written. However, the price was steep: Arnbjorn was trapped inside a crystal ball, unable to leave or interact with the world around him. He could only observe and learn, but never touch or feel.

At first, Arnbjorn was content with his newfound knowledge. He spent centuries studying the works of scholars, mages, and poets, and he became the greatest repository of knowledge in all of Tamriel. However, as time passed, Arnbjorn realized the true cost of his deal. He was cursed to spend eternity trapped in his crystal prison, unable to ever leave or be free. He became bitter and resentful, yearning for the life he once had.

Now, Arnbjorn serves as a cautionary tale, a warning to those who seek knowledge at any cost. He remains trapped in his crystal prison, a living embodiment of the dangers of making deals with Daedric princes.
'''


async def countTo10():
  await asyncio.sleep(10)
  print("Completed to 10")


async def askArnbjorn(query):
  print(f"QUERY HANDLER >> Asking Arnbjorn {query}")
  try:
    f = openai.Embedding.create(model="text-embedding-ada-002", input=query)
    query_embedding = np.array(f['data'][0]['embedding'])

    similarity = []
    for arr in pickled_df['embedding'].values:
      similarity.extend(
        cosine_similarity(query_embedding.reshape(1, -1), arr.reshape(1, -1)))
    context_chunk = chunks[np.argmax(similarity)]

    query_to_send = "CONTEXT: " + " ".join(context_chunk) + "\n\n" + query
    response = openai.Completion.create(model="text-davinci-003",
                                        prompt=query_to_send,
                                        max_tokens=2000,
                                        temperature=0)
    print("QUERY HANDLER >> Response Successfully Recieved!")
    if(response['choices'][0]['text'] == ""):
      return (
        "Arnbjorn doesn't seem like he wants to answer that. Try again later.")
    else:
      return (response['choices'][0]['text'][1:].strip())
  except:
    return (
      "Arnbjorn doesn't seem like he wants to answer that. Try again later.")
