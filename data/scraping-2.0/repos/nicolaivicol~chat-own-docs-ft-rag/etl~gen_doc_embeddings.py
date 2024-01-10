import pandas as pd
import requests
import os
import time
import random
from pathlib import Path
import logging
from tqdm import tqdm
import json
from typing import List, Dict, Any, Union, Tuple
import openai

import config
from etl.gen_docs_from_raw_jsons import list_all_codes_simple_json, load_doc_from_simple_json

log = logging.getLogger(os.path.basename(__file__))

openai.api_key = os.getenv("OPENAI_API_KEY", config.OPENAI_API_KEY)


def gen_embedding(code, use_title=True):
    doc = load_doc_from_simple_json(code)

    input = ''
    if use_title:
        input += doc['title']

    response = openai.Embedding.create(
        input=input,
        model=config.MODEL_EMBEDDING,
    )

    embedding = response["data"][0]["embedding"]

    return embedding


if __name__ == '__main__':
    log.info(f'Running {os.path.basename(__file__)}')
    doc_codes = list_all_codes_simple_json()

    for code in tqdm(doc_codes):
        time.sleep(0.1)
        e = gen_embedding(code, use_title=True)
        with open(f'{config.DIR_DATA}/services_embedding_title/{code}.json', 'w') as f:
            json.dump(e, f, ensure_ascii=False)

    log.info('Done.')



