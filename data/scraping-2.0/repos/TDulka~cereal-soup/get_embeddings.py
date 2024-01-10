import openai
import pandas as pd

openai.api_key_path = './key'

def get_embedding(text, model="text-embedding-ada-002"):
   return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']

birds = pd.read_csv('birds.csv')
birds['ada_embedding'] = birds['word'].apply(lambda x: get_embedding(x, model='text-embedding-ada-002'))
birds.to_csv('embedded_birds.csv')

soups = pd.read_csv('soups.csv')
soups['ada_embedding'] = soups['word'].apply(lambda x: get_embedding(x, model='text-embedding-ada-002'))
soups.to_csv('embedded_soups.csv')