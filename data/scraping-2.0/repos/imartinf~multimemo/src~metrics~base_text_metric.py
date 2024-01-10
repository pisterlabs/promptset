import click
import numpy as np
import openai
from sentence_transformers import SentenceTransformer
from openai.embeddings_utils import get_embedding

class BaseTextMetric:
    """
    Base class for text metrics that involve extracting embeddings using a model and then calculating a metric.
    """

    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def get_embeddings(self, texts, *args, **kwargs):
        """
        Extract embeddings from a text using the model and tokenizer.
        """
        if isinstance(self.model, SentenceTransformer):
            return self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True ,device=self.device)
        elif self.model == "precomputed" and 'path' in kwargs:
             return np.load(kwargs['path'], allow_pickle=True)
        elif isinstance(self.model, str):
            if isinstance(texts, list):
                click.warning(f"Warning: {self.model} only supports one text at a time.")
                click.warning("Using the first text in the list.")
                texts = texts[0]
            texts = texts.replace("\n", " ")
            response = openai.Embedding.create(input=texts, engine=self.model)['data'][0]['embedding']
            return np.array(response).reshape(1, -1)
        else:
            raise NotImplementedError
    
    def get_metric(self, texts1, texts2=None, labels=None):
        """
        Calculate the metric between two texts.
        """
        raise NotImplementedError
