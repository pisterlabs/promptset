import pandas as pd
import numpy as np
import sqlalchemy as sa
import itertools
from tqdm import tqdm
import openai

Secret = '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'
openai.organization = "org-j6fGVx3OgjgpAbCQFHOmdEUe"
openai.api_key = Secret


def load_measures(connection, condition_id):
  measures_query = 'select measures.* from measures join study_conditions on measures.study = study_conditions.study where study_conditions.condition = {}'
  measures = pd.read_sql(measures_query.format(condition_id), connection)

  return measures


def chunks(iterable, batch_size=100):
  """A helper function to break an iterable into chunks of size batch_size."""
  it = iter(iterable)
  chunk = tuple(itertools.islice(it, batch_size))
  while chunk:
    yield chunk
    chunk = tuple(itertools.islice(it, batch_size))


def get_embedding_openai(texts, model="text-embedding-ada-002"):
  texts = [x.replace("\n", " ") for x in texts]
  # print(texts)
  return [x['embedding'] for x in openai.Embedding.create(input = texts, model=model)['data']]


def encode_measures(measures, model='OpenAI'):
  measure_titles = measures[['title']].drop_duplicates()
  encoded = []
  if model == 'OpenAI':

    # Make sure this has the correct shape
    for titles in tqdm([list(c) for c in chunks(measure_titles['title'], batch_size=100)]):
      encoded += get_embedding_openai(titles)

    measure_titles = measure_titles.reset_index()
    measure_titles['vector'] = pd.Series(encoded)

  return measure_titles


def measure_emb_workflow(connection):
  measures = load_measures(connection, 369)

  encoded_titles = encode_measures(measures)

  encoded_titles.to_pickle('encoded_titles.pkl')


