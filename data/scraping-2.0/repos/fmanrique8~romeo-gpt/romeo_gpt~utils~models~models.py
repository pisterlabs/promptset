# romeo-gtp/romeo_gpt/utils/models.py

import openai
import numpy as np
import warnings
from . import embedding_model, completion_model, max_tokens


def get_embedding(text, api_key, model=embedding_model):
    openai.api_key = api_key
    try:
        if text is None or len(text) == 0:
            return None

        input_list = [text]
        response = openai.Embedding.create(input=input_list, model=model)
        return np.array(response["data"][0]["embedding"])
    except Exception as e:
        warnings.warn(f"Embedding failed for text: {text}. Error: {str(e)}")
        return None


def get_completion(prompt, api_key, model=completion_model):
    openai.api_key = api_key
    try:
        response = openai.Completion.create(
            engine=model,
            prompt=prompt,
            max_tokens=max_tokens,
            n=1,
            stop=None,
            temperature=0.5,
        )
        return response.choices[0].text.strip()
    except Exception as e:
        warnings.warn(f"Completion failed for prompt: {prompt}. Error: {str(e)}")
        return None
