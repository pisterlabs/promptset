import pandas as pd
import torch
from openai.embeddings_utils import get_embedding
numbers = [str(x) for x in range(1000)]
embeddings = [(num, get_embedding(num, "babbage-similarity")) for num in numbers]


df = pd.DataFrame(embeddings)

print(df)
df.to_pickle("numbers.pkl")
