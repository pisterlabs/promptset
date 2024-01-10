from dotenv import load_dotenv, find_dotenv
import json
import os
import openai
from openai.embeddings_utils import get_embedding, cosine_similarity
import numpy as np

_ = load_dotenv('openAI.env')
openai.api_key  = os.environ['openAI_api_key']

with open('movie_descriptions_embeddings.json', 'r') as file:
    file_content = file.read()
    movies = json.loads(file_content)

#Esta función devuelve una representación numérica (embedding) de un texto, en este caso
#la descripción de las películas
emb = get_embedding(movies[1]['description'],engine='text-embedding-ada-002')
print(emb)

#Vamos a crear una nueva llave con el embedding de la descripción de cada película en el archivo .json

'''
for i in range(len(movies)):
  emb = get_embedding(movies[i]['description'],engine='text-embedding-ada-002')
  movies[i]['embedding'] = emb


#Vamos a almacenar esta información en un nuevo archivo .json
with open('movie_descriptions_embeddings.json', 'r') as file:
    file_content = file.read()
    movies = json.loads(file_content)
'''
print(movies[0])

#Para saber cuáles películas se parecen más, podemos hacer lo siguiente:
print(movies[27]['title'])
print(movies[3]['title'])
print(movies[20]['title'])

#Calculamos la similitud de coseno entre los embeddings de las descripciones de las películas. Entre más alta la similitud
#más parecidas las películas.

print(f"Similitud entre película {movies[27]['title']} y {movies[3]['title']}: {cosine_similarity(movies[27]['embedding'],movies[3]['embedding'])}")
print(f"Similitud entre película {movies[27]['title']} y {movies[20]['title']}: {cosine_similarity(movies[27]['embedding'],movies[20]['embedding'])}")
print(f"Similitud entre película {movies[20]['title']} y {movies[3]['title']}: {cosine_similarity(movies[20]['embedding'],movies[3]['embedding'])}")

#Si se tuviera un prompt por ejemplo: Película de la segunda guerra mundial, podríamos generar el embedding del prompt y comparar contra 
#los embeddings de cada una de las películas de la base de datos. La película con la similitud más alta al prompt sería la película
#recomendada.

req = "película de la segunda guerra mundial"
emb = get_embedding(req,engine='text-embedding-ada-002')

sim = []
for i in range(len(movies)):
  sim.append(cosine_similarity(emb,movies[i]['embedding']))
sim = np.array(sim)
idx = np.argmax(sim)
print(movies[idx]['title'])


