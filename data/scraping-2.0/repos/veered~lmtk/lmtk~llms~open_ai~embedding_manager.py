from dataclasses import dataclass
import openai
import numpy as np # openai already depends on this

from .utils import load_api_key

vec_dim = 1536

class GPTEmbeddingManager:

  def __init__(self, model='text-embedding-ada-002'):
    load_api_key()

    self.model = model

    self.sources = []
    self.embeddings = []

  def calculate_embeddings(self, texts):
    if len(texts) == 0:
      return []

    # replace newlines, which can negatively affect performance.
    texts = [ text.replace('\n', ' ') for text in texts ]

    result = openai.Embedding.create(
      input=texts,
      model=self.model,
    )
    return [ r['embedding'] for r in result['data'] ]

  def calculate_embedding(self, text):
    return self.calculate_embeddings([ text ])[0]

  def add(self, *args, **kwargs):
    if isinstance(args[0], str):
      self.add_one(*args, **kwargs)
    else:
      self.add_multiple(*args, **kwargs)

  def add_one(self, text, meta=None, embedding=None):
    meta = {} if meta is None else meta

    self.sources.append({
      'index': len(self.embeddings),
      'text': text,
      'meta': meta,
    })

    if embedding == None:
      self.embeddings.append(self.calculate_embedding(text))
    else:
      self.embeddings.append(embedding)

  def add_multiple(self, inputs):
    for (i, input) in enumerate(inputs):
      if not isinstance(input, tuple):
        inputs[i] = (input, {})

    texts = [ text for (text, _) in inputs ]
    embeddings = self.calculate_embeddings(texts)

    for (i, embedding) in enumerate(embeddings):
      self.add(texts[i], meta=inputs[i][1], embedding=embedding)

  def search(self, text, top_n=5):
    if len(self.embeddings) == 0:
      return []

    embedding = self.calculate_embedding(text)
    similarities = np.matmul(self.embeddings, embedding)

    # similarities = np.matmul(np.asmatrix(manager.embeddings), embedding)
    # similarities = np.asarray(similarities)[0]

    top_indices = similarities.argsort()[::-1][:top_n]
    return [
      SearchResult(similarity=similarities[i], **self.sources[i])
      for i in top_indices
    ]

@dataclass
class SearchResult:
  similarity : float
  index: int
  text: str
  meta: dict
