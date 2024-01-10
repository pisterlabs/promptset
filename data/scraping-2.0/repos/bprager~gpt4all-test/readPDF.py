# #!/usr/bin/env python3

import logging
import os

from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import GPT4AllEmbeddings
from langchain.vectorstores.pgvector import PGVector
from PyPDF2 import PdfReader

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(name)s - %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def main():
    # Get the PDF file
    pdf = "helloworld.pdf"

    # extract the text
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    logging.debug(f"Text: {text}")

    # split into chunks
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
    )
    chunks = text_splitter.split_text(text)
    logging.debug(f"Chunks: {chunks}")

    # create embeddings
    embeddings = GPT4AllEmbeddings()
    # PGVector needs the connection string to the database.
    CONNECTION_STRING = "postgresql+psycopg2://postgres:postgres@localhost:5432/postgres"  # NOSONAR (Disable SonarLint warning)

    knowledge_base = PGVector.from_texts(
        connection_string=CONNECTION_STRING,
        embedding=embeddings,
        texts=chunks,
    )

    logging.debug(f"Knowledge base: {knowledge_base}")


if __name__ == "__main__":
    main()
