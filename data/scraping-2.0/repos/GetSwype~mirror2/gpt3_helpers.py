import os
import re
from time import time, sleep

import numpy as np
import openai
from numpy.linalg import norm

from constants import BOT_NAME
from utils import save_file


def gpt3_embedding(content, engine='text-embedding-ada-002'):
    """
    Returns a 512-dimensional vector embedding of the content.

    :param content: the content to embed
    :param engine: the GPT-3 engine to use
    :return: an N-dimensional vector embedding of the content
    """
    content = content.encode(encoding='ASCII', errors='ignore').decode()
    response = openai.Embedding.create(input=content, engine=engine)
    vector = response['data'][0]['embedding']  # this is a normal list
    return vector


def vector_similarity(v1, v2):
    """Returns the cosine similarity between two vectors.

    based upon https://stackoverflow.com/questions/18424228/cosine-similarity-between-2-number-lists
    """
    return np.dot(v1, v2) / (norm(v1) * norm(v2))  # return cosine similarity


def gpt3_completion(
        prompt,
        engine='text-davinci-003',
        temp=0.0,
        top_p=1.0,
        tokens=400,
        freq_pen=0.0,
        pres_pen=0.0,
        stop=None
) -> str:
    if stop is None:
        stop = ['USER:', f'{BOT_NAME}:']
    max_retry = 5
    retry = 0
    prompt = prompt.encode(encoding='ASCII', errors='ignore').decode()
    while True:
        try:
            response = openai.Completion.create(
                engine=engine,
                prompt=prompt,
                temperature=temp,
                max_tokens=tokens,
                top_p=top_p,
                frequency_penalty=freq_pen,
                presence_penalty=pres_pen,
                stop=stop
            )
            text = response['choices'][0]['text'].strip()
            text = re.sub('[\r\n]+', '\n', text)
            text = re.sub('[\t ]+', ' ', text)
            filename = '%s_gpt3.txt' % time()
            if not os.path.exists('gpt3_logs'):
                os.makedirs('gpt3_logs')
            save_file('gpt3_logs/%s' % filename, prompt + '\n\n==========\n\n' + text)
            return text
        except Exception as oops:
            retry += 1
            if retry >= max_retry:
                return "GPT3 error: %s" % oops
            print('Error communicating with OpenAI:', oops)
            sleep(1)
