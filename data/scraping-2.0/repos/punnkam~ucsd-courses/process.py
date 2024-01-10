from openai.embeddings_utils import get_embedding, cosine_similarity
from dotenv import load_dotenv
import openai
import pandas as pd
import numpy as np
import os

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

datafile_path = "./data/courses.csv"
df = pd.read_csv(datafile_path)
df = df.dropna()
# df = df[df.apply(lambda x: isinstance(x, str), axis=1)]
df['combined'] = "Name: " + df.name.str.strip() + "; Description: " + df.description.str.strip()

df['ada_search'] = df.combined.apply(lambda x: get_embedding(x, engine='text-embedding-ada-002'))
df.to_csv('./data/courses_processed.csv')