import os

from datasets import load_dataset
from elasticsearch import Elasticsearch
from openai.embeddings_utils import get_embedding
from tqdm.auto import tqdm

from elasticsearch_ir_evaluator.evaluator import (Document,
                                                  ElasticsearchIrEvaluator,
                                                  QandA)


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
    documents = []
    for i, row in enumerate(tqdm(corpus_dataset)):
        if i >= 100:
            break
        embedding = get_embedding(row["text"], engine="text-embedding-ada-002")
        documents.append(
            Document(
                id=row["docid"], title=row["title"], text=row["text"], vector=embedding
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
        if i >= 100:
            break
        vector = get_embedding(row["query"], engine="text-embedding-ada-002")

        qa_pairs.append(
            QandA(
                question=row["query"],
                answers=[p["docid"] for p in row["positive_passages"]],
                negative_answers=[p["docid"] for p in row["negative_passages"]],
                vector=vector,
            )
        )

    evaluator.load_qa_pairs(qa_pairs)

    # Set the index name in Elasticsearch
    # evaluator.set_index_name("corpus_xxxxxx")

    # Define a custom query template for Elasticsearch
    search_template = {
        "query": {
            "bool": {
                "must": {"match": {"text": "{{question}}"}},
            }
        },
        "knn": {
            "field": "vector",
            "query_vector": "{{vector}}",
            "k": 10,
            "num_candidates": 100,
        },
    }
    evaluator.set_search_template(search_template)

    # Calculate and print the Mean Reciprocal Rank (MRR)
    mrr = evaluator.calculate_mrr()
    print(f"MRR: {mrr}")


if __name__ == "__main__":
    main()
