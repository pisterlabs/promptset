import openai
import os
import numpy as np


class OpenAITokenizer:

    DEFAULT_MODEL = "text-embedding-ada-002"
    
    def __init__(self, key, modelName = None):
        openai.api_key = os.getenv(key)

        if modelName is None:
            self._model = OpenAITokenizer.DEFAULT_MODEL
        else:
            self._model = modelName

    def tokenize(self, inputs, max_requests = 500, max_tokens = 8192):

        if len(inputs) > max_requests:
            return np.concatenate([self.tokenize(inputs[i:i+max_requests], max_requests=max_requests, max_tokens=max_tokens) for i in range(0, len(inputs), max_requests)])

        #assume 3 letters per token
        if any([len(x) > max_tokens*3 for x in inputs]):
            return self.tokenize([x[:max_tokens*3] for x in inputs], max_requests=max_requests, max_tokens=max_tokens)

        response = openai.Embedding.create(
            model=self._model,
            input=inputs)
        
        return np.array([x['embedding'] for x in response['data']])