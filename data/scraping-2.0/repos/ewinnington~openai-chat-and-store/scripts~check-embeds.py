# pip install openai matplotlib plotly pandas scipy scikit-learn

import openai
from openai.embeddings_utils import get_embedding, cosine_similarity
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

#### If using azure, you'll need to adapt this
# openai.api_type = "azure"
# openai.api_key = AZURE_OPENAI_API_KEY
# openai.api_base = AZURE_OPENAI_ENDPOINT
# openai.api_version = "2022-12-01"
###

x1 = "dog"
x2 = "astronaut"

e1 = get_embedding(x1, engine = 'text-embedding-ada-002')
e2 = get_embedding(x2, engine = 'text-embedding-ada-002')

print(cosine_similarity(e1,e2))
## 0.80373316598824