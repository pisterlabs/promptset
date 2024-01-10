# import openai
import os
from dotenv import load_dotenv
import numpy as np
from typing import List, Optional
import asyncio
from asyncio import Semaphore
import logging
import openai
from time import time, sleep


if __name__ == '__main__':
    dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
    load_dotenv(dotenv_path)


openai.api_type = "azure"
openai.api_key = os.getenv("OPENAI_API_KEY")
if openai.api_key is None:
    logging.warning("openai.api_key is None")

openai.api_base = "https://test-oai69420.openai.azure.com/"
openai.api_version = "2023-05-15"
start = None
num_requests = 0

class TokenBucket:
    def __init__(self, rate: int):
        # rate is in requests per second
        self._rate = rate
        self._capacity = rate
        self._tokens = self._capacity
        self._last_refill = time()

    async def consume(self):
        while self._tokens < 1:
            self._refill()
            await asyncio.sleep(1)  # Sleep for some time before trying again
        self._tokens -= 1
        global num_requests
        num_requests += 1

    def _refill(self):
        now = time()
        time_passed = now - self._last_refill
        refill_amount = time_passed * self._rate
        self._tokens = min(self._capacity, self._tokens + refill_amount)
        self._last_refill = now

MaybeTokenBucket = Optional[TokenBucket]

async def _embed_sentence(sentence: str, token_bucket: MaybeTokenBucket=None) -> List[float]:
    done = False
    while not done:
        try:
            if token_bucket is not None:
                await token_bucket.consume()
            response = await openai.Embedding.acreate(
                input=sentence,
                engine="embeddings"
            )
            embedding = response['data'][0]['embedding']
            done = True
        except Exception as e:
            if token_bucket is None:
                sleep(1)
            print(e)
    return embedding


def embed_sentence(sentence: str) -> List[float]:
    return asyncio.run(_embed_sentence(sentence))


async def _handle_sentence(sentence: str, token_bucket: TokenBucket, semaphore: Semaphore) -> List[float]:
    async with semaphore:
        embedding = await _embed_sentence(sentence, token_bucket)
    global num_requests
    if num_requests % 1000 == 0:
        duration = time() - start
        duration_min = duration / 60
        print(f"{num_requests=}, {duration=:.2f} rate per min={num_requests / duration_min:.2f}")
    return embedding


def aembed_sentences(sentences: List[str]) -> List[List[float]]:
    global start
    start = time()
    global num_requests
    num_requests = 0
    max_concurrent_tasks = 20
    azure_quota_per_minute = 1000
    azure_quota_per_second = azure_quota_per_minute // 60
    semaphore = Semaphore(max_concurrent_tasks)
    token_bucket = TokenBucket(azure_quota_per_second)
    async def gather_tasks():
        tasks = [_handle_sentence(sentence, token_bucket, semaphore) for sentence in sentences]
        return await asyncio.gather(*tasks)
    return gather_tasks()

def embed_sentences(sentences: List[str]) -> List[List[float]]:
    global start
    start = time()
    global num_requests
    num_requests = 0
    max_concurrent_tasks = 20
    azure_quota_per_minute = 1000
    azure_quota_per_second = azure_quota_per_minute // 60
    semaphore = Semaphore(max_concurrent_tasks)
    token_bucket = TokenBucket(azure_quota_per_second)
    async def gather_tasks():
        tasks = [_handle_sentence(sentence, token_bucket, semaphore) for sentence in sentences]
        return await asyncio.gather(*tasks)
    return asyncio.run(gather_tasks())

async def _call_chat(system_prompt: str, user_prompt:str) -> str:
    messages= [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    # TODO: get configs
    response = await openai.ChatCompletion.acreate(
            engine="gpt-35-turbo-16k",
            messages=messages,
            )
    completion = response.choices[0].message
    return completion

def call_chat(system_prompt: str, user_prompt:str) -> str:
    messages= [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    # print(messages)
    # TODO: get configs
    response = openai.ChatCompletion.create(
            engine="gpt-35-turbo-16k",
            messages=messages,
            temperature=0,
            )
    completion = response.choices[0].message.content
    return completion

def cosine_similarity(query_embedding, document_embeddings):
    # assumes everything is normalized
    return document_embeddings @ query_embedding

def argknn(query_embedding, document_embeddings, k=10):
    dots = cosine_similarity(query_embedding, document_embeddings)
    pivot = len(dots) - k
    good_idxes = np.argpartition(dots, pivot)[-k:]
    good_dots = dots[good_idxes]
    sorted_good_idxes = good_idxes[np.argsort(good_dots)[::-1]]
    return sorted_good_idxes

def test_embeddings():
    sentence = "George Biden was the 46th President of the United States"
    embedding = embed_sentence(sentence)
    print(f"Embeddings are truncated to 5 elts")
    print(f"{sentence=}")
    print(f"{embedding[:5]=}")
    sentences = [sentence,
                "Old MacDonald had a farm EIEIEO",
                "The quick brown fox jumped over the lazy dog"]
    embeddings = embed_sentences(sentences)
    print('BATCHED')
    for s, e in zip(sentences, embeddings):
        print(f"sentence={s}")
        print(f"embedding={e[:5]}")


def test_cosine_similarity():
    documents = ["George Biden was the 46th President of the United States",
                "Old MacDonald had a farm EIEIEO",
                "The quick brown fox jumped over the lazy dog"]
    query = "And on that farm there was a dog"
    query_embedding = embed_sentence(query)
    document_embeddings = embed_sentences(documents)
    cosine_sims = cosine_similarity(query_embedding, document_embeddings)
    print(f"{documents=}")
    print(f"{cosine_sims=}")
    k = 2
    top_k_idxes = argknn(query_embedding, document_embeddings, k=k)
    top_k_docs = [documents[i] for i in top_k_idxes]
    print(f"{query=}")
    print(f"Top {k} documents: {top_k_docs}")


if __name__ == '__main__':
    test_embeddings()
    test_cosine_similarity()
