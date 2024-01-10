import os
import sys
import json
import argparse
import pandas as pd
from itertools import groupby
from typing import Dict, List
import concurrent.futures

from setup.embedding_dump import deflate_embeddings_map
from setup.setup_constants import EMBEDDING_FILE_NAME, HOTEL_REVIEW_FILE_NAME

from utils.ai import EMBEDDING_DIMENSION
from utils.db import init_cassio
from utils.reviews import format_review_content_for_embedding, get_review_vectorstore


# We create an ad-hoc "Embeddings" class, sitting on the precalculated embeddings,
# to perform all these insertions idiomatically through the LangChain
# abstraction. This is to avoid having to work at the bare-CassIO level
# while still taking advantage of the stored json with precalculated vectors.
from langchain.embeddings.base import Embeddings


class JustPreCalculatedEmbeddings(Embeddings):
    def __init__(self, precalc_dict: Dict[str, List[float]]) -> None:
        self.precalc_dict = precalc_dict

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_query(txt) for txt in texts]

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        if text in self.precalc_dict:
            return self.precalc_dict[text]
        else:
            # this happens from LangChain when creating the store:
            print(f"** [JustPreCalculatedEmbeddings] INFO: embed request for '{text}'. Returning moot results")
            return [0.0] * EMBEDDING_DIMENSION

    async def aembed_query(self, text: str) -> List[float]:
        return self.embed_query(text)


# This script stores the textual data and its embeddings into a table in the database.
# Note: this is a low-level, direct database interaction using cassIO to pre-populate the table. The API uses LangChain.
#
# It expects the following prerequisites:
#  - A vector-enabled database such as AstraDB.
#  - The hotel review CSV file generated in step 0.
#  - The compressed JSON file containing the precalculated embeddings (you can either use the precalculated embeddings
#      in this repo, or run step 1 to calculate them on the fly).
#
# The data is inserted asynchronously in batches within a given hotel to reduce loading time.
# Also, the hotels are processed concurrently in a thread-based way for the same reason.

this_dir = os.path.abspath(os.path.dirname(__file__))

DEFAULT_BATCH_SIZE = 100
DEFAULT_CONCURRENT_HOTELS = 80

if __name__ == "__main__":
    #
    parser = argparse.ArgumentParser(
        description="Store reviews with embeddings to cassIO vector table"
    )
    parser.add_argument(
        "-b",
        metavar="BATCH_SIZE",
        type=int,
        help="Batch size (for concurrent writes)",
        default=DEFAULT_BATCH_SIZE,
    )
    parser.add_argument(
        "-c",
        metavar="CONCURRENT_HOTELS",
        type=int,
        help="Number of hotels inserted at once",
        default=DEFAULT_CONCURRENT_HOTELS,
    )
    args = parser.parse_args()

    init_cassio()

    embedding_file_path = os.path.join(this_dir, EMBEDDING_FILE_NAME)
    if os.path.isfile(embedding_file_path):
        # review_id -> vector, which was stored in a compressed format to shrink file size
        enrichment = deflate_embeddings_map(json.load(open(embedding_file_path)))
    else:
        enrichment = {}

    hotel_review_file_path = os.path.join(this_dir, HOTEL_REVIEW_FILE_NAME)
    hotel_review_data = pd.read_csv(hotel_review_file_path)

    # sadly the precalc map for this "embeddings" must be sentence -> vector,
    # so we need a 'join' (which amounts to a preprocess pass through the hotel reviews dataframe)
    precalc_text_to_vector_map = {
        format_review_content_for_embedding(
            title=row["title"], body=row["text"]
        ): enrichment[row["id"]]
        for _, row in hotel_review_data.iterrows()
        if row["id"] in enrichment
    }
    c_embeddings = JustPreCalculatedEmbeddings(precalc_dict=precalc_text_to_vector_map)

    review_vectorstore = get_review_vectorstore(
        embeddings=c_embeddings,
        is_setup=True,
    )

    eligibles = (
        {
            "text": format_review_content_for_embedding(
                title=row["title"], body=row["text"]
            ),
            "metadata": {
                "hotel_id": row["hotel_id"],
                "rating": row["rating"],
                "title": row["title"],
            },
            "id": row["id"],
            "partition_id": row["hotel_id"],
        }
        for _, row in hotel_review_data.iterrows()
        if row["id"] in enrichment
    )

    def _flush_batch(store, batch):
        if batch:
            # collapse the arguments to lists: {"texts": [...], "ids": [... etc
            texts, metadatas, ids, partition_ids = list(
                zip(
                    *(
                        (eli["text"], eli["metadata"], eli["id"], eli["partition_id"])
                        for eli in batch
                    )
                )
            )
            # sanity check:
            assert len(set(partition_ids)) == 1
            # we need to group by partition_id and do separate inserts
            review_vectorstore.add_texts(
                texts=texts, metadatas=metadatas, ids=ids, partition_id=partition_ids[0]
            )
        return len(batch)

    groups_by_partition_id = groupby(
        sorted(eligibles, key=lambda eli: eli["partition_id"]),
        key=lambda eli: eli["partition_id"],
    )
    insertion_hotel_groups = [
        (part_id, list(items_in_hotel))
        for (part_id, items_in_hotel) in groups_by_partition_id
    ]

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.c) as executor:
        def handle_hotel(insertion_hotel_group):
            inserted = 0
            partition_id, items_in_partition_id = insertion_hotel_group
            items_list = list(items_in_partition_id)
            print(f"[{len(items_list)}] ", end="")
            sys.stdout.flush()
            # Even within a hotel, we might need to batch insertions:
            this_batch = []
            for eli in items_list:
                this_batch.append(eli)
                if len(this_batch) >= args.b:
                    # the batch is full: flush, then increment inserted counter
                    inserted += _flush_batch(review_vectorstore, this_batch)
                    this_batch = []
            # flush any insertions that may be left, then increment inserted counter
            if this_batch:
                inserted += _flush_batch(review_vectorstore, this_batch)
            this_batch = []
            return inserted

        print(f"[2-populate-review-vector-table.py] Inserting hotel reviews...")
        total_inserted = sum(executor.map(handle_hotel, insertion_hotel_groups))

    print(f"\n[2-populate-review-vector-table.py] Finished. {total_inserted} rows written.")
