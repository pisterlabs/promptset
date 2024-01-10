import sys
import os
from os.path import abspath, dirname
import logging
from langchain.embeddings.openai import OpenAIEmbeddings

os.environ["OPENAI_API_KEY"] = ""
embeddings = OpenAIEmbeddings(model='text-embedding-ada-002')

BASE_DIR = dirname(dirname(abspath(__file__)))  # /orbitkit
sys.path.insert(0, BASE_DIR)

from orbitkit.pdf_embedding.pdf_txt_embedding import PdfTxtEmbedding

logging.basicConfig(level=logging.INFO)


def embedding_pdf_from_s3(s3_path: str):
    pdf_extractor = PdfTxtEmbedding(s3_path=s3_path, embeddings=embeddings)
    pdf_extractor.embed()


if __name__ == "__main__":
    embedding_pdf_from_s3(s3_path="example.pdf")
    pass
