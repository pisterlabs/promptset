# %%
from models.open_ai import OpenAPI_search
import openai
from openai.embeddings_utils import get_embedding, cosine_similarity
from tqdm import tqdm
from common.constants import CSR_PATH
from common.sample_texts import GOOGLE_CSR_RAND_TEXTS
from common.SECRETS import API_KEY
import pandas as pd

# %%
openai.api_key = API_KEY

qry = "GHG emissions"
# %%
em_qry = get_embedding(qry, engine=OpenAPI_search.EMBEDDING_QRY_MODEL)
em_texts = [get_embedding(x, engine=OpenAPI_search.EMBEDDING_DOC_MODEL) for x in tqdm(GOOGLE_CSR_RAND_TEXTS)]

# %%
# Getting similarity between query and text snippet by embedding vector
similarity = [cosine_similarity(em_text, em_qry) for em_text in tqdm(em_texts)]
print(similarity)
# %%
