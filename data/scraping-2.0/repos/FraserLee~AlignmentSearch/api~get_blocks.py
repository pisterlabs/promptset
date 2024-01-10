from typing import List, Tuple
import dataclasses
import datetime
import itertools
import numpy as np
import openai
import regex as re
import requests
import time

# ---------------------------------- constants ---------------------------------

EMBEDDING_MODEL = "text-embedding-ada-002"

# ------------------------------------ types -----------------------------------

@dataclasses.dataclass
class Block:
    title: str
    author: str
    date: str
    url: str
    tags: str
    text: str
    
# ------------------------------------------------------------------------------

# Get the embedding for a given text. The function will retry with exponential backoff if the API rate limit is reached, up to 4 times.
def get_embedding(text: str) -> np.ndarray:

    max_retries = 4
    max_wait_time = 10
    attempt = 0

    while True:
        try:
            result = openai.Embedding.create(model=EMBEDDING_MODEL, input=text)
            return result["data"][0]["embedding"]

        except openai.error.RateLimitError as e:

            attempt += 1

            if attempt > max_retries: raise e

            time.sleep(min(max_wait_time, 2 ** attempt))


# Get the k blocks most semantically similar to the query using Pinecone.
def get_top_k_blocks(index, user_query: str, k: int) -> List[Block]:

    # Default to querying embeddings from live website if pinecone url not
    # present in .env
    #
    # This helps people getting started developing or messing around with the
    # site, since setting up a vector DB with the embeddings is by far the
    # hardest part for those not already on the team.

    if index is None:

        print('Pinecone index not found, performing semantic search on alignmentsearch-api.up.railway.app endpoint.')
        response = requests.post(
            "https://alignmentsearch-api.up.railway.app/semantic",
            json = {
                "query": user_query,
                "k": k
            }
        )

        return [Block(**block) for block in response.json()]

    # print time
    t = time.time()

    # Get the embedding for the query.
    query_embedding = get_embedding(user_query)

    t1 = time.time()
    print("Time to get embedding: ", t1 - t)

    query_response = index.query(
        namespace="alignment-search",  # ugly, sorry
        top_k=k,
        include_values=False,
        include_metadata=True,
        vector=query_embedding
    )
    blocks = []
    for match in query_response['matches']:

        date = match['metadata']['date']

        if type(date) == datetime.date: date = date.strftime("%Y-%m-%d") # iso8601

        blocks.append(Block(
            title = match['metadata']['title'],
            author = match['metadata']['author'],
            date = date,
            url = match['metadata']['url'],
            tags = match['metadata']['tags'],
            text = strip_block(match['metadata']['text'])
        ))

    t2 = time.time()

    print("Time to get top-k blocks: ", t2 - t1)
    
    # for all blocks that are "the same" (same title, author, date, url, tags),
    # combine their text with "....." in between. Return them in order such
    # that the combined block has the minimum index of the blocks combined.

    key = lambda bi: (bi[0].title or "", bi[0].author or "", bi[0].date or "", bi[0].url or "", bi[0].tags or "")

    blocks_plus_old_index = [(block, i) for i, block in enumerate(blocks)]
    blocks_plus_old_index.sort(key=key)

    unified_blocks: List[Tuple[Block, int]] = []

    for key, group in itertools.groupby(blocks_plus_old_index, key=key):
        group = list(group)
        if len(group) == 0: continue
        
        group = group[:3] # limit to a max of 3 blocks from any one source

        text = "\n.....\n".join([block[0].text for block in group])

        min_index = min([block[1] for block in group])

        unified_blocks.append((Block(key[0], key[1], key[2], key[3], key[4], text), min_index))

    unified_blocks.sort(key=lambda bi: bi[1])
    return [block for block, _ in unified_blocks]


# we add the title and authors inside the contents of the block, so that
# searches for the title or author will be more likely to pull it up. This
# strips it back out.
def strip_block(text: str) -> str:
    r = re.match(r"^\"(.*)\"\s*-\s*Title:.*$", text, re.DOTALL)
    if not r:
        print("Warning: couldn't strip block")
        print(text)
    return r.group(1) if r else text
