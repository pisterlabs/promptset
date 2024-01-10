import logging

import click

from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
import torch
from utils import load_documents, split_documents

from constants import DEVICE_TYPES, SOURCE_DIRECTORY, EMBEDDING_MODEL_NAME


@click.command()
@click.option(
    "--device_type",
    default="cuda" if torch.cuda.is_available() else "cpu",
    type=click.Choice(
        DEVICE_TYPES,
    ),
    help="Device to run on. (Default is cuda)",
)
def main(device_type):
    # Load documents and split in chunks
    logging.info(f"Loading documents from {SOURCE_DIRECTORY}")

    embeddings = HuggingFaceInstructEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": device_type},
        # embed_instruction="Represent the Javascript software library document for retrieval:",
    )

    tokenizer = embeddings.client.tokenizer

    documents = load_documents(SOURCE_DIRECTORY)
    texts = split_documents(documents, tokenizer=None)

    logging.info(f"Loaded {len(documents)} documents from {SOURCE_DIRECTORY}")
    logging.info(f"Split into {len(texts)} chunks of text")

    if texts:
        db = FAISS.from_documents(
            texts,
            embedding=embeddings,
        )
        bindata = db.serialize_to_bytes()
        with open("DB/faiss.pickke", "wb") as f:
            f.write(bindata)

        query = "What is Volto?"
        docs = db.similarity_search(query)
        assert len(docs) > 0
    else:
        logging.warning("No documents found to be indexed")


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s",
        level=logging.INFO,
    )
    main()
