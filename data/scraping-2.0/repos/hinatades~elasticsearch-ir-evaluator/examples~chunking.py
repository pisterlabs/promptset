import os

from datasets import load_dataset
from elasticsearch import Elasticsearch
from openai.embeddings_utils import get_embedding
from tqdm.auto import tqdm

from elasticsearch_ir_evaluator.evaluator import (Document,
                                                  ElasticsearchIrEvaluator,
                                                  Passage, QandA)


def sliding_window_text_chunking(text, window_size, overlap):
    """
    Splits a given text into chunks using a sliding window approach.

    :param text: The input text to be chunked.
    :param window_size: The size of each window/chunk in characters.
    :param overlap: The number of characters to overlap between consecutive chunks.
    :return: A list of text chunks.
    """
    # Ensure the window size and overlap are valid
    if window_size <= 0 or overlap < 0 or overlap >= window_size:
        raise ValueError("Invalid window size or overlap value.")

    chunks = []
    start = 0
    while start < len(text):
        # Determine the end position of the current chunk
        end = start + window_size
        # Add the chunk to the list
        chunks.append(text[start:end])
        # Move the start position for the next chunk, taking overlap into account
        start = end - overlap

    return chunks


def main():
    # Initialize Elasticsearch client using environment variables
    es_client = Elasticsearch(
        hosts=os.environ["ES_HOST"],
        basic_auth=(os.environ["ES_USERNAME"], os.environ["ES_PASSWORD"]),
        verify_certs=False,
        ssl_show_warn=False,
    )

    # Create an instance of ElasticsearchIrEvaluator
    evaluator = ElasticsearchIrEvaluator(es_client)

    # Load the corpus dataset and vectorize the text of each document
    corpus_dataset = load_dataset(
        "castorini/mr-tydi-corpus",
        "japanese",
        split="train",
        trust_remote_code=True,
    )

    window_size = 200
    overlap = 20

    documents = []
    for i, row in enumerate(tqdm(corpus_dataset)):
        passages = [
            Passage(text=sentence, vector=None)
            for sentence in sliding_window_text_chunking(
                row["text"], window_size, overlap
            )
        ]
        documents.append(
            Document(
                id=row["docid"],
                title=row["title"],
                text=row["text"],
                passages=passages,
            )
        )
    evaluator.load_corpus(documents)

    evaluator.create_index_from_corpus()
    evaluator.index_corpus()

    # Load the QA dataset and vectorize each query
    qa_dataset = load_dataset(
        "castorini/mr-tydi",
        "japanese",
        split="test",
        trust_remote_code=True,
    )
    qa_pairs = []
    for i, row in enumerate(tqdm(qa_dataset)):
        qa_pairs.append(
            QandA(
                question=row["query"],
                answers=[p["docid"] for p in row["positive_passages"]],
                negative_answers=[p["docid"] for p in row["negative_passages"]],
            )
        )

    evaluator.load_qa_pairs(qa_pairs)

    # Define a custom query template for Elasticsearch
    search_template = {
        "query": {
            "nested": {
                "path": "passages",
                "query": {"match": {"passages.text": "{{question}}"}},
                "inner_hits": {},
            },
        }
    }
    evaluator.set_search_template(search_template)

    # Calculate and print the Mean Reciprocal Rank (MRR)
    mrr = evaluator.calculate_mrr()
    print(f"MRR: {mrr}")


if __name__ == "__main__":
    main()
