import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import GPT4AllEmbeddings
from analyzer import analyze

import utils

import re
import time

DATA_PATH = "data/"
DB_PATH = "vectorstores/db/"

load_dotenv()
LANG = os.getenv("LANGUAGE")


def create_vector_db():
    text_translator = utils.create_translator()

    loader = PyPDFDirectoryLoader(DATA_PATH)
    documents = loader.load()
    print(f"Processed {len(documents)} pdf files")
    exponential_backoff = 1
    for idx, doc in enumerate(documents):
        content = re.sub(r'\s+', ' ', doc.page_content)
        content = content.replace('\n', '').replace('\r', '')
        content = re.sub(r'^[0-9]+\s', '', content, flags=re.MULTILINE)

        if not content.strip():
            documents.remove(doc)
            continue

        try:
            translated_page_content = utils.translate_message(
                text_translator, content, LANG, 'en', True)

            if exponential_backoff >= 1:
                if exponential_backoff > 1:
                    exponential_backoff /= 2
                else:
                    exponential_backoff = 0
                print(f"Exponential backoff: {exponential_backoff}")

        except Exception as exception:
            error_code = exception.args[0]
            if error_code == 429001 and exponential_backoff < 16:
                if exponential_backoff == 0:
                    exponential_backoff = 1
                else:
                    exponential_backoff *= 2
                print(f"Exponential backoff: {exponential_backoff}")
            translated_page_content = ""

        doc.page_content = translated_page_content
        if idx % 10 == 0:
            print(f"Translated {idx} documents")
        time.sleep(exponential_backoff)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    vectorstore = Chroma.from_documents(
        documents=texts, embedding=GPT4AllEmbeddings(), persist_directory=DB_PATH)

    common_words, entities_ro, topic_words = analyze()

    vectorstore.add_texts(
        common_words, embedding=GPT4AllEmbeddings(), persist_directory=DB_PATH)
    vectorstore.add_texts(
        entities_ro, embedding=GPT4AllEmbeddings(), persist_directory=DB_PATH)
    vectorstore.add_texts(
        topic_words, embedding=GPT4AllEmbeddings(), persist_directory=DB_PATH)

    vectorstore.persist()


if __name__ == "__main__":
    create_vector_db()
