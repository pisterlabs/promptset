import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from typing import List

import openai
from sqlalchemy import delete
from tenacity import retry, wait_random_exponential, stop_after_attempt

from app import babotree_utils
from app.database import get_direct_db
from app.models import Highlight, ContentEmbedding, SourceType

openai.api_key = babotree_utils.get_secret("OPENAI_API_KEY")


@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def get_embedding(text: str, model="text-embedding-ada-002") -> list[float]:
    return openai.embeddings.create(input=text, model=model).data[0].embedding


def embed_highlight(highlight):
    print('embedding highlight', highlight.id)
    text_to_embed = highlight.text
    try:
        embedding = get_embedding(text_to_embed)
    except Exception as e:
        print('error getting embedding for highlight:', highlight.id, e)
        time.sleep(60)
        return
    db = get_direct_db()
    highlight_vector_embedding = ContentEmbedding(
        source_id=highlight.id,
        source_type=SourceType.HIGHLIGHT_TEXT.value,
        embedding=embedding,
    )
    print("removing old embedding for highlight:", highlight.id)
    db.execute(delete(ContentEmbedding).where(ContentEmbedding.source_id == highlight.id, ContentEmbedding.source_type == SourceType.HIGHLIGHT_TEXT.value))
    print('adding new embedding for highlight:', highlight.id)
    db.add(highlight_vector_embedding)
    db.commit()
    db.close()
    print('done embedding highlight:', highlight.id)


def embed_highlights_from_queue(local_queue):
    while True:
        print("waiting for highlight to embed")
        try:
            highlight = local_queue.get()
            if highlight is None:
                # None is termination signal
                print("got termination signal, exiting")
                break
            embed_highlight(highlight)
            local_queue.task_done()
        except Exception as e:
            print("error getting highlight from queue", e)
def start_producer_consumer(highlight_ids_to_embed: List[uuid.UUID]):
    queue = Queue()
    num_consumers = 25
    with ThreadPoolExecutor(max_workers=num_consumers) as executor:
        print("starting consumers")
        for i in range(num_consumers):
            executor.submit(embed_highlights_from_queue, queue)
        print("starting producer")
        chunk_size = 100
        db = get_direct_db()
        for i in range(0, len(highlight_ids_to_embed), chunk_size):
            chunk = highlight_ids_to_embed[i:i + chunk_size]
            print("adding chunk to queue")
            highlights_to_embed = db.query(Highlight).filter(Highlight.id.in_(chunk)).all()
            for highlight in highlights_to_embed:
                queue.put(highlight)
        print("Finished adding chunks to queue, adding termination signals")
        for i in range(num_consumers):
            queue.put(None)
        db.close()
        print("Producer done")


def main():
    # we want to pull all highlights from the db and embed the ones
    # that don't have embeddings
    db = get_direct_db()
    # start by identifying which highlights don't have embeddings
    highlight_ids = db.query(Highlight.id).all()
    highlight_ids = [x[0] for x in highlight_ids]
    existing_highlight_embeddings_ids = db.query(ContentEmbedding.source_id)\
        .filter(ContentEmbedding.source_type == SourceType.HIGHLIGHT_TEXT.value)\
        .all()
    db.close()
    existing_highlight_embeddings_ids = set([x[0] for x in existing_highlight_embeddings_ids])
    highlight_ids_to_embed = [x for x in highlight_ids if x not in existing_highlight_embeddings_ids]
    print(f"Found {len(highlight_ids_to_embed)} highlights to embed")
    # now we want to embed the text of each highlight
    start_producer_consumer(highlight_ids_to_embed)



if __name__ == '__main__':
    main()