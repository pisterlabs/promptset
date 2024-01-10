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
from app.models import Highlight, ContentEmbedding, SourceType, HighlightSource

openai.api_key = babotree_utils.get_secret("OPENAI_API_KEY")


@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def get_embedding(text: str, model="text-embedding-ada-002") -> list[float]:
    return openai.embeddings.create(input=text, model=model).data[0].embedding


def get_all_highlights_from_source(highlight_source):
    db = get_direct_db()
    highlights = db.query(Highlight).filter(Highlight.source_id == highlight_source.id).all()
    db.close()
    return "\n---\n".join([highlight.text for highlight in highlights])


def embed_highlight_source(highlight_source):
    print('embedding highlight source', highlight_source.id)
    text_to_embed = get_all_highlights_from_source(highlight_source)
    try:
        embedding = get_embedding(text_to_embed)
    except Exception as e:
        print('error getting embedding for highlight source:', highlight_source.id, e)
        time.sleep(60)
        return
    db = get_direct_db()
    highlight_vector_embedding = ContentEmbedding(
        source_id=highlight_source.id,
        source_type=SourceType.ALL_HIGHLIGHTS_FROM_SOURCE.value,
        embedding=embedding,
    )
    print("removing old embedding for highlight:", highlight_source.id)
    db.execute(delete(ContentEmbedding).where(ContentEmbedding.source_id == highlight_source.id, ContentEmbedding.source_type == SourceType.ALL_HIGHLIGHTS_FROM_SOURCE.value))
    print('adding new embedding for highlight source:', highlight_source.id)
    db.add(highlight_vector_embedding)
    db.commit()
    db.close()
    print('done embedding highlight source:', highlight_source.id)


def embed_highlight_source_from_queue(local_queue):
    while True:
        print("waiting for highlight source to embed")
        try:
            highlight_source = local_queue.get()
            if highlight_source is None:
                # None is termination signal
                print("got termination signal, exiting")
                break
            embed_highlight_source(highlight_source)
            local_queue.task_done()
        except Exception as e:
            print("error getting highlight source from queue", e)
def start_producer_consumer(highlight_source_ids_to_embed: List[uuid.UUID]):
    queue = Queue()
    num_consumers = 25
    with ThreadPoolExecutor(max_workers=num_consumers) as executor:
        print("starting consumers")
        for i in range(num_consumers):
            executor.submit(embed_highlight_source_from_queue, queue)
        print("starting producer")
        chunk_size = 100
        db = get_direct_db()
        for i in range(0, len(highlight_source_ids_to_embed), chunk_size):
            chunk = highlight_source_ids_to_embed[i:i + chunk_size]
            print("adding chunk to queue")
            highlight_sources_to_embed = db.query(HighlightSource).filter(HighlightSource.id.in_(chunk)).all()
            for highlight_source in highlight_sources_to_embed:
                queue.put(highlight_source)
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
    highlight_source_ids = db.query(HighlightSource.id).all()
    highlight_source_ids = [x[0] for x in highlight_source_ids]
    existing_highlight_source_embeddings_ids = db.query(ContentEmbedding.source_id)\
        .filter(ContentEmbedding.source_type == SourceType.ALL_HIGHLIGHTS_FROM_SOURCE.value)\
        .all()
    db.close()
    existing_highlight_source_embeddings_ids = set([x[0] for x in existing_highlight_source_embeddings_ids])
    highlight_source_ids_to_embed = [x for x in highlight_source_ids if x not in existing_highlight_source_embeddings_ids]
    print(f"Found {len(highlight_source_ids_to_embed)} highlights to embed")
    # now we want to embed the text of each highlight
    start_producer_consumer(highlight_source_ids_to_embed)



if __name__ == '__main__':
    main()