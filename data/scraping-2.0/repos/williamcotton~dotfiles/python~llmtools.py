import re
from openai import OpenAI
import json
import tqdm
from typing import List, Tuple, Dict, Any

client = OpenAI()

TOKEN_LIMIT = 8192

def embedding(input_text):
    response = client.embeddings.create(
        input=input_text,
        model="text-embedding-ada-002"
    )
    embedding = response.data[0].embedding
    return embedding

def sanitize_json_string(json_string):
    # Remove invalid control characters
    sanitized_string = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', json_string)
    return sanitized_string

def get_title(text):
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=[
            {"role": "system", "content": """Find the document title and return JSON: { "title": %%TITLE%% }"""},
            {"role": "user", "content": text[:300]}
        ],
        response_format={ "type": "json_object" }
    )

    # Return the model's message
    json_string = completion.choices[0].message.content
    sanitized_string = sanitize_json_string(json_string)
    try:
        title_object = json.loads(sanitized_string)
        title = title_object['title']
    except Exception as e:
        return False
    return title


def batch_embeddings(keysAndTextTuple: Tuple[List[str], List[str]]) -> List[Tuple[str, List[float]]]:
    keys: List[str]
    texts: List[str]
    keys, texts = zip(*keysAndTextTuple)
    batches: List[List[Tuple[str, str]]] = []
    batch: List[Tuple[str, str]] = []
    token_count: int = 0
    for key, text in zip(keys, texts):
        if token_count + len(text) > TOKEN_LIMIT:
            batches.append(batch)
            batch = [(key, text)]
            token_count = len(text)
        else:
            batch.append((key, text))
            token_count += len(text)

    if batch:
        batches.append(batch)

    embeddings: List[Tuple[str, List[float]]] = []
    for batch in tqdm.tqdm(batches, desc="Processing LLM embeddings batches", unit="batch"):
        keys, texts = zip(*batch)
        response: Dict[str, Any] = client.embeddings.create(
            input=texts,
            model="text-embedding-ada-002"
        )
        for key, embedding in zip(keys, response.data):
            embeddings.append((key, embedding.embedding))

    return embeddings