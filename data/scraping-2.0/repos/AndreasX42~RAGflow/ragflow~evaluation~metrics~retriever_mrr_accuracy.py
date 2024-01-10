import asyncio
import numpy as np
from langchain.schema.vectorstore import VectorStoreRetriever
from langchain.schema.document import Document

import logging

logger = logging.getLogger(__name__)


async def grade_retriever(
    label_dataset: list[dict], retrieverForGrading: VectorStoreRetriever
) -> tuple[float, float, float]:
    """Calculate metrics for the precision of the retriever using MRR (Mean Reciprocal Rank) scores.
    For every list of retrieved documents we calculate the rank of the chunks wrt reference chunk from
    the label dataset.

    Args:
        label_dataset (list[dict]): _description_
        retrieved_docs (list[str]): _description_
        retrieverForGrading (VectorStoreRetriever): _description_

    Returns:
        tuple[float,float,float]: MRR scores for top 3, 5 and 10
    """

    logger.info("Grading retrieved document chunks using MRR.")

    # get predicted answers with top 10 retrieved document chunks
    retrieved_chunks = await asyncio.gather(
        *[
            retrieverForGrading.aget_relevant_documents(qa_pair["question"])
            for qa_pair in label_dataset
        ]
    )

    ref_chunks = [
        Document(page_content=label["metadata"]["context"], metadata=label["metadata"])
        for label in label_dataset
    ]

    return calculate_mrr(ref_chunks, retrieved_chunks)


def calculate_mrr(ref_chunks, retrieved_chunks):
    """Calculates mrr scores."""

    top3, top5, top10 = [], [], []

    # Check to ensure ref_chunks is not empty
    if not ref_chunks:
        return 0, 0, 0

    for ref_chunk, retr_chunks in zip(ref_chunks, retrieved_chunks):
        hit = False
        for idx, chunk in enumerate(retr_chunks, 1):
            if is_hit(ref_chunk, chunk):
                rank = 1 / idx
                if idx <= 3:
                    top3.append(rank)
                if idx <= 5:
                    top5.append(rank)
                if idx <= 10:
                    top10.append(rank)
                hit = True
                break

        if not hit:  # If there's no hit, the rank contribution is 0
            top3.append(0)
            top5.append(0)
            top10.append(0)

    # Calculate the MRR for the top 3, 5, and 10 documents
    mrr_3 = sum(top3) / len(ref_chunks)
    mrr_5 = sum(top5) / len(ref_chunks)
    mrr_10 = sum(top10) / len(ref_chunks)

    return mrr_3, mrr_5, mrr_10


def is_hit(ref_chunk, retrieved_chunk):
    """Checks if retrieved chunk is close to reference chunk."""

    # Check if both chunks are from same document
    if ref_chunk.metadata["source"] != retrieved_chunk.metadata["source"]:
        return False

    label_start, label_end = (
        ref_chunk.metadata["start_index"],
        ref_chunk.metadata["end_index"],
    )
    retr_start, retr_end = (
        retrieved_chunk.metadata["start_index"],
        retrieved_chunk.metadata["end_index"],
    )

    retr_center = retr_start + (retr_end - retr_start) // 2

    # Consider retrieved chunk to be a hit if it is near the reference chunk
    return retr_center in range(label_start, label_end + 1)
