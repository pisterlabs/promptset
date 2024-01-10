import openai
from .utils import load_api_key

class GPTEmbedder:

  vec_dim = 1536

  def __init__(self, model='text-embedding-ada-002'):
    load_api_key()
    self.model = model

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
