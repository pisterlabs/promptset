# GPT3 API

import openai
import numpy as np
import pandas as pd
import json
openai.api_key = ""


def get_embedding(texts, model="text-embedding-ada-002"):
    ret = np.empty((0, 1536))
    for text in texts:
        text.replace("\n", " ")
        np.vstack((ret, openai.Embedding.create(input = texts, model=model)['data'][0]['embedding']))
    return ret

lines = []
with open('./mini.json', 'r') as f:
    for x in f:
        lines.append(json.loads(x))
df = pd.DataFrame(lines)

