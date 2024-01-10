import argparse
from itertools import islice
from pathlib import Path

from tqdm import tqdm
import torch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Qdrant

from loaders import get_loader, LOADER_NAMES
from config import DB_CONFIG


CHUNK_SIZE = 500


def get_text_chunk(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=0
    )
    texts = text_splitter.split_documents(docs)
    return texts


def batched(iterable, *, size=100):
    "Batch data into tuples of length n. The last batch may be shorter."
    # batched('ABCDEFG', 3) --> ABC DEF G
    if size < 1:
        raise ValueError('n must be at least one')
    it = iter(iterable)
    while batch := tuple(islice(it, size)):
        yield batch


def store(texts):
    model_name = "intfloat/multilingual-e5-large"
    model_kwargs = {"device": "cuda:0" if torch.cuda.is_available() else "cpu"}
    encode_kwargs = {"normalize_embeddings": False}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )
    db_url, db_api_key, db_collection_name = DB_CONFIG
    for batch in tqdm(batched(texts, size=100)):
        _ = Qdrant.from_documents(
            batch,
            embeddings,
            url=db_url,
            api_key=db_api_key,
            collection_name=db_collection_name,
        )


def get_parser():
    p = argparse.ArgumentParser()
    p.add_argument("index", type=str)
    p.add_argument("inputfile", metavar="INPUTFILE", type=str)
    p.add_argument("-l", "--loader", type=str, choices=LOADER_NAMES, required=True)
    return p


def index_annotated_docs(docs, index):
    for doc in docs:
        doc.metadata["index"] = index
        yield doc


def main():
    """
    $ python store.py --loader wikipage "index" "FILE_PATH"
    $ python store.py -l wikipage wiki data/wiki.json
    $ python store.py -l rtdhtmlpage django ./docs.djangoproject.com/
    """
    p = get_parser()
    args = p.parse_args()
    loader = get_loader(
        args.loader,
        inputfile=Path(args.inputfile),
    )

    docs = loader.lazy_load()
    texts = get_text_chunk(index_annotated_docs(docs, args.index))
    store(texts)


if __name__ == "__main__":
    main()
