from sentence_transformers import SentenceTransformer
import cohere
import numpy as np
import voyageai
from voyageai import get_embeddings
from tqdm.notebook import tqdm
from config import COHERE_KEY, VOYAGEAI_KEY

model_names = [
    'SentenceTransformer/bert-base-nli-mean-tokens',
    'cohere-english',
    'cohere-multilingual',
    'voyageai',  # very slow
    # TODO: add more models
]

# this are trial key
cohere_key = COHERE_KEY
voyageai.api_key = VOYAGEAI_KEY


def embed(model_name, sentences):
    if model_name not in model_names:
        raise ValueError(f'Invalid model name: {model_name}')
    
    match model_name:
        case 'SentenceTransformer/bert-base-nli-mean-tokens':
            model = SentenceTransformer('bert-base-nli-mean-tokens')
            return model.encode(sentences)
        case 'cohere-english':
            co = cohere.Client(cohere_key)
            embs = co.embed(sentences, input_type='clustering', model='embed-english-v3.0').embeddings
            return np.array(embs)
        case 'cohere-multilingual':
            co = cohere.Client(cohere_key)
            embs = co.embed(sentences, input_type='clustering', model='embed-multilingual-v3.0').embeddings
            return np.array(embs)
        case 'voyageai':
            max_batch_size = 8 # constraint from voyageai
            embs = []
            for i in tqdm(range(0, len(sentences), max_batch_size)):
                embs += get_embeddings(sentences[i:i+max_batch_size])
            return np.array(embs)
