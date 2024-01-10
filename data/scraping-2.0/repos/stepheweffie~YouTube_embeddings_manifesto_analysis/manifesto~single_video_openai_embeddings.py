import pandas as pd
from nltk.tokenize import sent_tokenize
import os
from manifesto_openai_embeddings import openai_embedding_call, count_tokens
from dotenv import load_dotenv
import numpy as np

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')

data = pd.read_pickle(f'single_video_transcript.pkl')
data = pd.DataFrame(data)
data = data['text']
sents = list()

# Preprocess sentences
data = data.apply(lambda x: x.replace('\n', ' '))
data_list = data.apply(lambda x: sent_tokenize(text=x))
# Convert Series to a list for count_embedding_tokens
data = data_list.tolist()
for sents_lists in data:
    sents_lists = sents_lists[0]
    sents.append(sents_lists)
# Get tokens as a dataframe and call OpenAI API for embeddings on encoded tokens


tokens = count_tokens(sents)
# Do something else with the tokens dataframe

text = tokens['encoded_tokens']
embedding_array = openai_embedding_call(text)

# Save as a pickle file data/single_video_openai_embeddings.pkl




