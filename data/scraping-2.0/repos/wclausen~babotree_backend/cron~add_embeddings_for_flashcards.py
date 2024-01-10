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
from app.models import Highlight, ContentEmbedding, SourceType, Flashcard

openai.api_key = babotree_utils.get_secret("OPENAI_API_KEY")


@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def get_embedding(text: str, model="text-embedding-ada-002") -> list[float]:
    return openai.embeddings.create(input=text, model=model).data[0].embedding


def embed_flashcard(flashcard):
    print('embedding flashcard', flashcard.id)
    text_to_embed = flashcard.question + "\n" + flashcard.answer
    try:
        embedding = get_embedding(text_to_embed)
    except Exception as e:
        print('error getting embedding for flashcard:', flashcard.id, e)
        time.sleep(60)
        return
    db = get_direct_db()
    flashcard_vector_embedding = ContentEmbedding(
        source_id=flashcard.id,
        source_type=SourceType.FLASHCARD_TEXT.value,
        embedding=embedding,
    )
    print("removing old embedding for flashcard:", flashcard.id)
    db.execute(delete(ContentEmbedding).where(ContentEmbedding.source_id == flashcard.id, ContentEmbedding.source_type == SourceType.FLASHCARD_TEXT.value))
    print('adding new embedding for flashcard:', flashcard.id)
    db.add(flashcard_vector_embedding)
    db.commit()
    db.close()
    print('done embedding flashcard:', flashcard.id)


def embed_flashcards_from_queue(local_queue):
    while True:
        print("waiting for flashcard to embed")
        try:
            flashcard = local_queue.get()
            if flashcard is None:
                # None is termination signal
                print("got termination signal, exiting")
                break
            embed_flashcard(flashcard)
            local_queue.task_done()
        except Exception as e:
            print("error getting flashcard from queue", e)
def start_producer_consumer(flashcard_ids_to_embed: List[uuid.UUID]):
    queue = Queue()
    num_consumers = 25
    with ThreadPoolExecutor(max_workers=num_consumers) as executor:
        print("starting consumers")
        for i in range(num_consumers):
            executor.submit(embed_flashcards_from_queue, queue)
        print("starting producer")
        chunk_size = 100
        db = get_direct_db()
        for i in range(0, len(flashcard_ids_to_embed), chunk_size):
            ids_chunk = flashcard_ids_to_embed[i:i + chunk_size]
            print("adding chunk to queue")
            flashcards_to_embed = db.query(Flashcard).filter(Flashcard.id.in_(ids_chunk)).all()
            for flashcard in flashcards_to_embed:
                queue.put(flashcard)
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
    flashcard_ids = db.query(Flashcard.id).all()
    flashcard_ids = [x[0] for x in flashcard_ids]
    existing_flashcard_embeddings_ids = db.query(ContentEmbedding.source_id)\
        .filter(ContentEmbedding.source_type == SourceType.FLASHCARD_TEXT.value)\
        .all()
    db.close()
    existing_flashcard_embeddings_ids = set([x[0] for x in existing_flashcard_embeddings_ids])
    flashcard_ids_to_embed = [x for x in flashcard_ids if x not in existing_flashcard_embeddings_ids]
    print(f"Found {len(flashcard_ids_to_embed)} highlights to embed")
    # now we want to embed the text of each highlight
    start_producer_consumer(flashcard_ids_to_embed)



if __name__ == '__main__':
    main()